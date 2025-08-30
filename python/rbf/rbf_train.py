import numpy as np
from scipy import signal as spsig
from dataclasses import dataclass
from typing import Optional, TypedDict
import pickle
from pathlib import Path

# MPEM binding (pybind11-backed) will be provided in rbf/mpem.py
try:
    from rbf.mpem import ActEngine  # noqa: F401
except ImportError:
    ActEngine = None
    # This is for compatibility with older code/notebooks if needed
    mpem_extract = None

class ActResultDict(TypedDict):
    """Typed dictionary for the results of an ACT extraction."""
    frequency: float
    chirp_rate: float
    amplitude: float
    duration: float
    time_center: float
    spectral_width: float

# -----------------------------------------------------------------------------
# Domain-specific constants for EEG focus detection (locked in)
# -----------------------------------------------------------------------------
EEG_FS: float = 256.0
ACT_WINDOW_SAMPLES: int = 256  # 1.0 s @ 256 Hz

# Tailored ACT dictionary bounds for high-beta/low-gamma bursts
EEG_ACT_RANGES = {
    "tc_min": 0.0,
    "tc_max": float(ACT_WINDOW_SAMPLES - 1),
    "tc_step": 8.0,
    "fc_min": 20.0,
    "fc_max": 45.0,
    "fc_step": 2.0,  # lean dictionary (~15k atoms)
    "logDt_min": -3.0,
    "logDt_max": -1.0,
    "logDt_step": 0.5,
    "c_min": -5.0,
    "c_max": 7.0,
    "c_step": 2.0,
}


def enforce_window(sig: np.ndarray, length: int = ACT_WINDOW_SAMPLES) -> np.ndarray:
    """Center-crop or zero-pad a 1D signal to the desired length.

    This ensures compatibility with a persistent ActEngine that expects
    a fixed-length window.
    """
    x = np.asarray(sig, dtype=np.float64).ravel()
    n = x.size
    if n == length:
        return x
    if n > length:
        # center crop
        start = (n - length) // 2
        return x[start:start + length]
    # zero-pad to the right
    out = np.zeros((length,), dtype=np.float64)
    out[:n] = x
    return out


def sliding_windows(sig: np.ndarray, fs: float = EEG_FS, window_s: float = 1.0, stride_s: float = 0.25) -> np.ndarray:
    """Return a stack of fixed-length windows with given stride (for inference)."""
    x = np.asarray(sig, dtype=np.float64).ravel()
    N = int(round(window_s * fs))
    step = max(1, int(round(stride_s * fs)))
    if x.size < N:
        return enforce_window(x, N)[None, :]
    segments = []
    for start in range(0, max(1, x.size - N + 1), step):
        segments.append(x[start:start + N])
        if start + N >= x.size:
            break
    return np.vstack(segments) if segments else enforce_window(x, N)[None, :]


class ActFeatureExtractor:
    """Reusable ACT feature extractor that keeps a persistent ACT dictionary.

    Falls back to one-shot mpem_extract if ActEngine is unavailable.
    """
    def __init__(
        self,
        fs: float = EEG_FS,
        length: int = ACT_WINDOW_SAMPLES,
        ranges: Optional[dict] = EEG_ACT_RANGES,
        *,
        complex_mode: bool = False,
        force_regenerate: bool = False,
        mute: bool = True,
        dict_cache_file: str = "dict_cache.bin",
    ) -> None:
        self.fs = float(fs)
        self.length = int(length)
        self.ranges = ranges
        self.dict_cache_file = dict_cache_file
        self.engine = None

        if ActEngine is not None:
            self.engine = ActEngine(
                self.fs,
                self.length,
                ranges=self.ranges if self.ranges is not None else None,
                complex_mode=complex_mode,
                force_regenerate=force_regenerate,
                mute=mute,
                dict_cache_file=dict_cache_file,
            )
        else:
            # This path is taken if the import failed, for clarity
            raise RuntimeError(
                "ACT bindings are unavailable. Build and install the pybind11 extension (rbf.mpem)."
            )

    def extract_dict(self, sig: np.ndarray, order: int = 1) -> ActResultDict:
        """Run the ACT transform and return features as a dictionary."""
        x = enforce_window(sig, self.length)
        if self.engine is not None:
            # Enable debug flag to activate C++ logging
            return self.engine.transform(x, order=order, debug=True)
        raise RuntimeError("ACT bindings are unavailable. Build and install the pybind11 extension (rbf.mpem).")

    def extract_feature_vector(self, sig: np.ndarray, order: int = 1) -> np.ndarray:
        """Run ACT and return features as a NumPy vector, handling NaNs."""
        feat = self.extract_dict(sig, order=order)
        
        # Extract features, defaulting to NaN for robustness
        fc = float(feat.get("frequency", np.nan))
        c = float(feat.get("chirp_rate", np.nan))
        amp = float(feat.get("amplitude", np.nan))
        duration = float(feat.get("duration", np.nan))
        tc = float(feat.get("time_center", np.nan))
        
        # Safely calculate spectral width
        if np.isnan(duration) or duration <= 1e-12:
            spec_w = np.nan
        else:
            spec_w = float(feat.get("spectral_width", 1.0 / (2 * np.pi * duration)))

        feature_vector = np.array([fc, c, amp, duration, tc, spec_w], dtype=np.float64)

        # If any critical feature is NaN, return a vector of NaNs
        # This allows the caller to easily filter out failed extractions.
        if np.isnan(feature_vector).any():
            return np.full_like(feature_vector, np.nan)
            
        return feature_vector


# Module-level default extractor (persistent ACT if available)
DEFAULT_EXTRACTOR: Optional[ActFeatureExtractor]
try:
    DEFAULT_EXTRACTOR = ActFeatureExtractor()
except Exception:
    # Defer error until first use
    DEFAULT_EXTRACTOR = None

class SyntheticBetaGammaGenerator:
    """Generate realistic beta/gamma bursts for mental math"""
    
    def __init__(self):
        self.fs = int(EEG_FS)  # Sampling frequency
        
    def generate_mental_math_burst(self, 
                                   frequency=25,  # Beta center
                                   chirp_rate=2,  # Hz/second upward
                                   duration=1.0,
                                   snr_db=-5):
        """Generate a single realistic burst"""
        
        t = np.arange(0, duration, 1/self.fs)
        
        # 1. Create chirplet (frequency increases during calculation)
        instantaneous_freq = frequency + chirp_rate * t
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.fs
        
        # 2. Apply realistic envelope (rise-sustain-decay)
        envelope = self._cognitive_envelope(t, duration)
        chirplet = envelope * np.sin(phase)
        
        # 3. Add realistic EEG background
        background = self._generate_eeg_background(len(t))
        
        # 4. Mix at specified SNR
        signal_power = np.mean(chirplet**2)
        noise_power = signal_power / (10**(snr_db/10))
        background = background * np.sqrt(noise_power / np.mean(background**2))
        
        synthetic_eeg = chirplet + background
        
        # 5. Add realistic artifacts occasionally
        if np.random.random() < 0.2:  # 20% chance
            synthetic_eeg += self._add_artifacts(synthetic_eeg)
        
        return synthetic_eeg, chirplet  # Return both for validation
    
    def _cognitive_envelope(self, t, duration):
        """Realistic cognitive burst envelope"""
        # Fast rise (100ms), sustain, slow decay (200ms)
        rise_time = 0.1
        decay_time = 0.2
        
        envelope = np.ones_like(t)
        
        # Rise phase
        rise_mask = t < rise_time
        envelope[rise_mask] = t[rise_mask] / rise_time
        
        # Decay phase
        decay_start = duration - decay_time
        decay_mask = t > decay_start
        envelope[decay_mask] = 1 - (t[decay_mask] - decay_start) / decay_time
        
        # Add slight amplitude modulation (realistic)
        envelope *= (1 + 0.1 * np.sin(2 * np.pi * 7 * t))  # 7 Hz tremor
        
        return envelope
    
    def _generate_eeg_background(self, n_samples):
        """Generate realistic 1/f EEG background (real-valued).
        Uses Hermitian-symmetric spectrum shaping and irfft.
        Adds a narrow alpha band bump around 10 Hz.
        """
        fs = self.fs
        # Use rfft domain (non-negative freqs)
        freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)
        # Avoid divide by zero at DC by adding 1.0
        mag = 1.0 / (np.maximum(freqs, 1.0) ** 1.0)
        # Random phase for positive frequency bins (except DC and Nyquist)
        phases = np.random.uniform(0.0, 2 * np.pi, size=freqs.shape)
        phases[0] = 0.0
        # If Nyquist exists (even n), set phase 0 to keep real signal
        if n_samples % 2 == 0:
            phases[-1] = 0.0
        spectrum = mag * np.exp(1j * phases)

        # Add an alpha bump (Gaussian around 10 Hz)
        alpha_center = 10.0
        alpha_width = 1.5  # Hz
        alpha_gain = 0.25
        alpha_bump = alpha_gain * np.exp(-0.5 * ((freqs - alpha_center) / alpha_width) ** 2)
        spectrum += alpha_bump

        # Inverse real FFT to obtain real-valued background
        background = np.fft.irfft(spectrum, n=n_samples)
        # Normalize to unit std
        std = np.std(background) if np.std(background) > 1e-8 else 1.0
        return background / std
    
    def _add_artifacts(self, sig):
        """Add realistic artifacts (eye blinks, EMG)."""
        artifact_type = np.random.choice(['blink', 'muscle'])

        if artifact_type == 'blink':
            # Eye blink: slow, large amplitude with random timing
            t = np.arange(len(sig)) / self.fs
            center = np.random.uniform(0.2, 0.8) * t[-1] if t[-1] > 0 else 0.5
            width = np.random.uniform(0.05, 0.12)  # 50–120 ms
            amp = np.random.uniform(2.0, 6.0)
            blink = amp * np.exp(-((t - center) ** 2) / (2 * width * width))
            return blink

        # EMG: band-limited high-frequency noise burst
        emg = np.random.randn(len(sig))
        b, a = spsig.butter(4, [20, 100], btype='bandpass', fs=self.fs)
        emg = spsig.filtfilt(b, a, emg)
        # Gate EMG to a short random window
        gate = np.zeros_like(emg)
        n = len(emg)
        if n > 10:
            w = max(5, int(0.05 * self.fs))
            start = np.random.randint(0, max(1, n - w))
            gate[start:start + w] = 1.0
        return 0.5 * emg * gate

    def generate_alpha_burst(self, duration: float = 1.0):
        """Generate an alpha burst (rest state, ~10 Hz)."""
        t = np.arange(0, duration, 1.0 / self.fs)
        env = self._cognitive_envelope(t, duration)
        sig = 0.8 * env * np.sin(2 * np.pi * 10.0 * t)
        bg = self._generate_eeg_background(len(t))
        return (sig + 0.7 * bg)

    def generate_motor_beta(self, duration: float = 1.0, freq: float = 20.0):
        """Generate a non-chirped beta burst (e.g., motor imagery)."""
        t = np.arange(0, duration, 1.0 / self.fs)
        env = self._cognitive_envelope(t, duration)
        sig = env * np.sin(2 * np.pi * freq * t)
        bg = self._generate_eeg_background(len(t))
        return (sig + 0.7 * bg)

class SyntheticTrainingDataset:
    def __init__(self, n_samples=10000):
        self.generator = SyntheticBetaGammaGenerator()
        self.n_samples = n_samples
        
    def create_labeled_dataset(self):
        """Create complete training dataset"""
        
        X_chirplets = []
        y_labels = []
        
        # 1. True Positives: Mental Math Patterns
        for i in range(self.n_samples // 2):
            # Vary parameters realistically
            freq = np.random.uniform(18, 35)  # Beta to gamma
            chirp = np.random.uniform(0, 5)   # Upward chirp
            duration = np.random.uniform(0.5, 2.0)
            snr = np.random.uniform(-10, 5)   # Realistic SNR range
            
            sig, true_chirp = self.generator.generate_mental_math_burst(
                frequency=freq,
                chirp_rate=chirp,
                duration=duration,
                snr_db=snr
            )
            
            # Extract chirplet features and only keep valid ones
            features = self.extract_chirplet_features(sig)
            if not np.isnan(features).any():
                X_chirplets.append(features)
                y_labels.append(1)  # Mental math
        
        # 2. True Negatives: Other EEG Patterns
        for i in range(self.n_samples // 4):
            # Generate alpha bursts (rest state)
            dur = np.random.uniform(0.5, 2.0)
            sig = self.generator.generate_alpha_burst(duration=dur)
            features = self.extract_chirplet_features(sig)
            if not np.isnan(features).any():
                X_chirplets.append(features)
                y_labels.append(0)  # Not mental math
        
        # 3. Hard Negatives: Similar but different patterns
        for i in range(self.n_samples // 4):
            # Motor imagery beta (similar frequency, no chirp)
            dur = np.random.uniform(0.5, 2.0)
            sig = self.generator.generate_motor_beta(duration=dur)
            features = self.extract_chirplet_features(sig)
            if not np.isnan(features).any():
                X_chirplets.append(features)
                y_labels.append(0)  # Not mental math
            
        return np.array(X_chirplets), np.array(y_labels)
    
    def extract_chirplet_features(self, sig):
        """Extract chirplet parameters via persistent ACT instance (fallback to shim)."""
        if DEFAULT_EXTRACTOR is None:
            raise RuntimeError(
                "ACT extractor unavailable. Build and install the pybind11 bindings (rbf.mpem)."
            )
        return DEFAULT_EXTRACTOR.extract_feature_vector(sig, order=1)

class RBFClassifier:
    """NumPy-based RBF classifier (CPU), keeping API compatibility."""
    def __init__(self, n_centers: int = 50, kmeans_iters: int = 50, seed: int = 0):
        self.n_centers = n_centers
        self.kmeans_iters = kmeans_iters
        self.seed = seed
        self.centers: Optional[np.ndarray] = None
        self.widths: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def _standardize_fit(self, X: np.ndarray):
        """Compute and store mean and std for standardization."""
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # Add a small epsilon to prevent division by zero for zero-variance features
        self.std_[self.std_ < 1e-8] = 1e-8

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """Standardize data using stored mean and std."""
        return (X - self.mean_) / self.std_

    def _kmeans_np(self, X: np.ndarray, k: int, iters: int) -> np.ndarray:
        """A simple, robust NumPy-based k-means implementation."""
        if len(X) == 0:
            # If there's no data, we can't determine centers.
            # Return an empty array with the correct shape.
            return np.empty((0, X.shape[1]), dtype=X.dtype)

        rng = np.random.default_rng(self.seed)
        # Initialize centers from data, ensuring we don't request more centers than points
        n_points = len(X)
        k = min(k, n_points)
        idx = rng.choice(n_points, size=k, replace=False)
        centers = X[idx].copy()
        for _ in range(iters):
            # Assign
            d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(d2, axis=1)
            # Update
            for j in range(k):
                members = X[labels == j]
                if len(members) > 0:
                    centers[j] = members.mean(axis=0)
        return centers

    def _compute_widths_np(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        # Width per center as median distance to its assigned points; fallback to global median
        d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        widths = np.zeros((centers.shape[0],), dtype=np.float64)
        global_d = np.sqrt(d2.min(axis=1))
        global_med = np.median(global_d) if global_d.size else 1.0
        for j in range(centers.shape[0]):
            dj = np.sqrt(d2[labels == j, j])
            widths[j] = np.median(dj) if dj.size else global_med
        # Prevent zeros
        widths[widths < 1e-6] = global_med if global_med > 0 else 1.0
        return widths

    def _design_matrix_np(self, X: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> np.ndarray:
        # Compute ||x-c||^2 efficiently
        X2 = np.sum(X * X, axis=1, keepdims=True)
        C2 = np.sum(centers * centers, axis=1, keepdims=True).T
        XC = X @ centers.T
        d2 = np.maximum(0.0, X2 + C2 - 2.0 * XC)
        H = np.exp(-d2 / (2.0 * (widths[None, :] ** 2)))
        return H

    def train_on_synthetic(self, X_synthetic: np.ndarray, y_synthetic: np.ndarray):
        """Train RBF entirely on synthetic data (CPU)."""
        X = np.asarray(X_synthetic, dtype=np.float64)
        y = np.asarray(y_synthetic, dtype=np.float64)

        # Standardize
        self._standardize_fit(X)
        Xz = self._standardize(X)

        # Centers from positives
        X_pos = Xz[y == 1]
        if len(X_pos) == 0:
            X_pos = Xz  # fallback
        k = min(self.n_centers, max(1, len(X_pos)))
        self.centers = self._kmeans_np(X_pos, k, self.kmeans_iters)

        # Widths
        self.widths = self._compute_widths_np(Xz, self.centers)

        # Train linear weights
        H = self._design_matrix_np(Xz, self.centers, self.widths)
        self.weights, *_ = np.linalg.lstsq(H, y, rcond=None)
        return self

    def predict_realtime(self, new_signal: np.ndarray, fs: float = 256.0) -> float:
        """Prediction on new EEG segment using persistent ACT-based features."""
        if DEFAULT_EXTRACTOR is None:
            raise RuntimeError("ACT extractor unavailable; build and install pybind11 bindings.")
        x = DEFAULT_EXTRACTOR.extract_feature_vector(np.asarray(new_signal, dtype=np.float64), order=1)[None, :]

        # Standardize with training stats
        xz = self._standardize(x)
        H = self._design_matrix_np(xz, self.centers, self.widths)
        pred = float((H @ self.weights).ravel()[0])
        return pred

# End of rbf_train module

def main():
    """Main function to run the RBF training pipeline."""
    print("🕵️  Running in debug mode to isolate a failing signal for C++...")

    # 1. Generate one synthetic signal
    generator = SyntheticBetaGammaGenerator()
    test_signal, _ = generator.generate_mental_math_burst(
        frequency=28, chirp_rate=4, snr_db=0
    )

    # 2. Save the signal and parameters to files
    signal_path = Path("debug_signal.txt")
    params_path = Path("debug_params.json")
    
    np.savetxt(signal_path, test_signal)
    print(f"Saved test signal to '{signal_path}'")

    # Save default ranges used by the extractor
    import json
    if DEFAULT_EXTRACTOR and DEFAULT_EXTRACTOR.ranges:
        with open(params_path, "w") as f:
            json.dump(DEFAULT_EXTRACTOR.ranges, f, indent=2)
        print(f"Saved ACT parameters to '{params_path}'")
    else:
        print("Could not access default ranges to save.")

    # 3. Run the transform on just this one signal to confirm failure
    print("\nRunning transform on the isolated signal...")
    if DEFAULT_EXTRACTOR:
        # Run it twice to check for state issues
        print("--- First call ---")
        features1 = DEFAULT_EXTRACTOR.extract_feature_vector(test_signal)
        print(f"Extracted features (1): {features1}")
        print("\n--- Second call ---")
        features2 = DEFAULT_EXTRACTOR.extract_feature_vector(test_signal)
        print(f"Extracted features (2): {features2}")
    
    print("\nDebug data saved. Exiting.")

if __name__ == "__main__":
    # Check if the ACT bindings are available before running
    if DEFAULT_EXTRACTOR is None:
        print("❌ Fatal Error: ACT bindings (rbf.mpem) are not available.")
        print("Please build and install the pybind11 extension first.")
    else:
        main()