from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from pathlib import Path
import numpy as np

from .act_dict_io import read_act_dict_header


@dataclass
class ActExtractorConfig:
    fs: float = 256.0
    length: int = 256
    ranges: Optional[Dict[str, float]] = None  # if None, binding defaults are used
    complex_mode: bool = False
    force_regenerate: bool = False
    mute: bool = True
    dict_cache_file: str = "act_mlp_cache.bin"
    order: int = 1  # top-1 by default for features


class ActFeatureExtractor:
    """Wrapper around pyact.mpbfgs.ActEngine to compute ACT features per window.

    Notes
    -----
    - The mpbfgs binding returns a dict with keys: 'params', 'coeffs', 'error', 'signal', 'approx', 'residue'
      where 'params' is a list of [tc, fc, logDt, c] per component and 'coeffs' the corresponding amplitudes.
    - This extractor computes a compact feature vector for order=1 by default:
        [fc_hz, chirp_hz_per_s, amplitude, duration_s, time_center_s, spectral_width_hz]
    - If transform fails or outputs are missing, returns NaNs.
    """

    def __init__(self, config: Optional[ActExtractorConfig] = None) -> None:
        if config is None:
            config = ActExtractorConfig()
        self.cfg = config

        try:
            from pyact.mpbfgs import ActEngine  # type: ignore
        except Exception as e:  # pragma: no cover - import path
            raise ImportError(
                "pyact.mpbfgs extension not available. Build and install the C++ bindings."
            ) from e

        # If a dictionary file path is provided and exists, read its header to
        # auto-sync fs/length with the persisted dictionary.
        if self.cfg.dict_cache_file:
            p = Path(self.cfg.dict_cache_file)
            if p.exists():
                h = read_act_dict_header(str(p))
                if h is not None:
                    self.cfg.fs = float(h.fs)
                    self.cfg.length = int(h.length)

        self._engine = ActEngine(
            self.cfg.fs,
            int(self.cfg.length),
            self.cfg.ranges if self.cfg.ranges is not None else None,
            self.cfg.complex_mode,
            self.cfg.force_regenerate,
            self.cfg.mute,
            self.cfg.dict_cache_file,
        )

    @property
    def fs(self) -> float:
        return float(self._engine.fs)

    @property
    def length(self) -> int:
        return int(self._engine.length)

    def extract_feature_vector(self, signal_1d: np.ndarray, order: Optional[int] = None) -> np.ndarray:
        """Compute ACT features for a single 1D window.

        If order==k, returns a concatenated vector of length 6*k:
          [fc_1, c_1, amp_1, dur_1, tc_s_1, specw_1, fc_2, c_2, ..., specw_k]
        Missing components (early stopping) are filled with NaNs; callers can drop such rows.
        """
        x = np.asarray(signal_1d, dtype=np.float64).ravel()
        if x.size != self.length:
            raise ValueError(f"Signal length {x.size} != engine length {self.length}")

        ord_use = int(self.cfg.order if order is None else order)

        try:
            out = self._engine.transform(x, order=ord_use, debug=False)
            params = out.get("params", [])
            coeffs = out.get("coeffs", [])
            if not params or not coeffs:
                return np.full((6 * ord_use,), np.nan, dtype=np.float64)

            vec = np.full((6 * ord_use,), np.nan, dtype=np.float64)
            use_n = min(ord_use, len(params), len(coeffs))
            for i in range(use_n):
                tc_samp, fc_hz, logDt, c_hz_s = [float(v) for v in params[i]]
                amp = float(coeffs[i])
                duration_s = float(np.exp(logDt))
                tc_s = tc_samp / float(self.fs)
                spectral_w = (1.0 / (2.0 * np.pi * duration_s)) if duration_s > 1e-12 else np.nan
                off = 6 * i
                vec[off:off+6] = [fc_hz, c_hz_s, abs(amp), duration_s, tc_s, spectral_w]
            return vec
        except Exception:
            return np.full((6 * ord_use,), np.nan, dtype=np.float64)

    def extract_features_batch(self, X: np.ndarray, channel: int = 0, order: Optional[int] = None) -> np.ndarray:
        """Compute features for a batch of windows.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, n_channels, n_times) or (n_windows, n_times)
        channel : int
            Channel index to use if X has multiple channels.
        order : Optional[int]
            Transform order; defaults to config.order.
        """
        X = np.asarray(X)
        if X.ndim == 3:
            # (N, C, T)
            if X.shape[2] != self.length:
                raise ValueError(f"Window length {X.shape[2]} != engine length {self.length}")
            x_iter = X[:, channel, :]
        elif X.ndim == 2:
            if X.shape[1] != self.length:
                raise ValueError(f"Window length {X.shape[1]} != engine length {self.length}")
            x_iter = X
        else:
            raise ValueError("X must be (n_windows, n_channels, n_times) or (n_windows, n_times)")

        # Determine feature dimensionality with first window
        first = self.extract_feature_vector(x_iter[0], order=order)
        feats = np.zeros((x_iter.shape[0], first.shape[0]), dtype=np.float64)
        feats[0] = first
        for i in range(1, x_iter.shape[0]):
            feats[i] = self.extract_feature_vector(x_iter[i], order=order)
        return feats
