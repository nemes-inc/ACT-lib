import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import cwt, morlet2

# Create a synthetic signal with interesting time-frequency content
def create_test_signal(t):
    """Create a signal with three distinct components:
    1. A pure tone at the beginning (440 Hz - A note)
    2. A chirp in the middle (sweeping from 200 to 800 Hz)
    3. Two simultaneous tones at the end (523 Hz and 659 Hz - C and E notes)
    """
    signal = np.zeros_like(t)
    
    # Component 1: Pure tone (0-1 second)
    mask1 = (t >= 0) & (t < 1)
    signal[mask1] = np.sin(2 * np.pi * 440 * t[mask1])
    
    # Component 2: Chirp (1-2 seconds)
    mask2 = (t >= 1) & (t < 2)
    t_chirp = t[mask2] - 1  # Reset time to start at 0
    # Frequency sweeps from 200 to 800 Hz
    instantaneous_freq = 200 + 600 * t_chirp
    phase = 2 * np.pi * (200 * t_chirp + 300 * t_chirp**2)
    signal[mask2] = np.sin(phase)
    
    # Component 3: Two simultaneous tones (2-3 seconds)
    mask3 = (t >= 2) & (t <= 3)
    signal[mask3] = (np.sin(2 * np.pi * 523 * t[mask3]) + 
                     np.sin(2 * np.pi * 659 * t[mask3])) / 2
    
    return signal

# Set up time array
fs = 2000  # Sampling frequency (Hz)
duration = 3  # Total duration (seconds)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
test_signal = create_test_signal(t)

# Create figure with subplots
fig, axes = plt.subplots(4, 1, figsize=(12, 12))

# 1. Plot the original signal
axes[0].plot(t, test_signal, 'b-', linewidth=0.5)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Original Signal: Pure Tone (0-1s) → Chirp (1-2s) → Chord (2-3s)')
axes[0].grid(True, alpha=0.3)

# 2. Short-Time Fourier Transform (STFT) / Spectrogram
f_stft, t_stft, Zxx = signal.stft(test_signal, fs, window='hann', 
                                   nperseg=256, noverlap=200)
axes[1].pcolormesh(t_stft, f_stft, np.abs(Zxx), shading='gouraud', cmap='hot')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('Method 1: STFT/Spectrogram (Fixed Window Size)')
axes[1].set_ylim([0, 1000])
axes[1].axvline(x=1, color='white', linestyle='--', alpha=0.5)
axes[1].axvline(x=2, color='white', linestyle='--', alpha=0.5)

# 3. Pseudo Wigner–Ville Distribution (proper FFT over lags)
# Using an analytic signal and a lag window to reduce cross-term artifacts
from scipy.signal import hilbert

def pseudo_wvd(x, fs, time_decim=4, lag_len=257):
    """Compute a pseudo Wigner–Ville distribution.

    Parameters
    ---------
    x : array-like
        Real-valued input signal.
    fs : float
        Sampling frequency in Hz.
    time_decim : int
        Decimation factor along time to speed up the computation.
    lag_len : int (odd)
        Number of lags (window length). Higher -> better frequency resolution.

    Returns
    -------
    W : ndarray, shape (lag_len//2 + 1, n_times)
        Time–frequency energy (non-negative freqs from rFFT of lag product).
    t_wvd : ndarray
        Time vector in seconds corresponding to W columns.
    f_wvd : ndarray
        Frequency vector in Hz corresponding to W rows.
    """
    N = len(x)
    x_a = hilbert(x)  # analytic signal -> mostly positive freqs

    # Ensure odd lag length
    if lag_len % 2 == 0:
        lag_len += 1
    half = (lag_len - 1) // 2

    # Time indices where we can place the lag window fully inside the signal
    times = np.arange(half, N - half, time_decim)

    # Lag window (Hamming) to form a pseudo-WVD and suppress cross-terms
    lag_window = np.hamming(lag_len)

    # Allocate output for non-negative frequency bins from rFFT
    W = np.zeros((half + 1, len(times)))

    # Pre-allocate buffer for the instantaneous autocorrelation sequence
    k = np.zeros(lag_len, dtype=complex)

    for ti, n in enumerate(times):
        # Build instantaneous autocorrelation for lags -half..+half
        # x[n+tau] * conj(x[n-tau])
        for tau in range(-half, half + 1):
            k[tau + half] = x_a[n + tau] * np.conj(x_a[n - tau]) * lag_window[tau + half]

        # FFT over lags -> frequency slice at time n
        S = np.fft.rfft(k)
        # WVD is real-valued for auto-terms of analytic signals (numerical noise aside)
        W[:, ti] = np.real(S)

    f_wvd = np.linspace(0, fs / 2, half + 1)
    t_wvd = times / fs
    return W, t_wvd, f_wvd

W, t_wigner, freq_wigner = pseudo_wvd(test_signal, fs, time_decim=4, lag_len=257)
axes[2].pcolormesh(t_wigner, freq_wigner, np.abs(W), shading='gouraud', cmap='hot')
axes[2].set_ylabel('Frequency (Hz)')
axes[2].set_xlabel('Time (s)')
axes[2].set_title('Method 2: (Simplified) Wigner Distribution')
axes[2].set_ylim([0, 1000])
axes[2].axvline(x=1, color='white', linestyle='--', alpha=0.5)
axes[2].axvline(x=2, color='white', linestyle='--', alpha=0.5)

# 4. Continuous Wavelet Transform (CWT)
# Modified wavelet transform section with correct scale↔frequency mapping
# Use linear frequency spacing for better visualization
frequencies = np.linspace(100, 1000, 128)
w0 = 6.0  # Morlet center frequency (nondimensional)
# For morlet2, center frequency f ≈ w0 * fs / (2π * s) -> s = w0 * fs / (2π f)
widths = (w0 * fs) / (2 * np.pi * frequencies)
cwt_matrix = cwt(test_signal, morlet2, widths, w=w0)

# Use log scale for better contrast
cwt_power = np.abs(cwt_matrix)**2
cwt_db = 10 * np.log10(cwt_power + 1e-10)  # Convert to dB scale

axes[3].pcolormesh(t, frequencies, cwt_db, 
                   shading='gouraud', cmap='hot',
                   vmin=np.percentile(cwt_db, 10),  # Adjust contrast
                   vmax=np.percentile(cwt_db, 99))

axes[3].set_ylabel('Frequency (Hz)')
axes[3].set_xlabel('Time (s)')
axes[3].set_title('Method 3: Wavelet Transform (Better Resolution Trade-off)')
axes[3].set_ylim([100, 1000])
axes[3].axvline(x=1, color='white', linestyle='--', alpha=0.5)
axes[3].axvline(x=2, color='white', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Print what to look for
print("\n🔍 What to observe in each representation:\n")
print("1. STFT/Spectrogram:")
print("   - Fixed time-frequency resolution (rectangular tiles)")
print("   - Good for stationary signals but blurry for chirps")
print("\n2. Wigner Distribution:")
print("   - Better resolution but can have interference artifacts")
print("   - Notice cross-terms between components")
print("\n3. Wavelet Transform:")
print("   - Variable resolution: narrow in time for high frequencies")
print("   - Better tracks the chirp's changing frequency")
print("\n📍 White dashed lines mark transitions between signal components")
print("\n✨ The chirplet transform (this paper) would adaptively fit")
print("   curved 'blobs' to match each component optimally!")