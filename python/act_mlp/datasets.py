from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class EdfWindowConfig:
    """Configuration for loading EDF/EDF+ files into fixed-length windows using MNE.

    Attributes
    ----------
    edf_path : str
        Path to the EDF/EDF+ file.
    pick_channels : Optional[List[str]]
        List of channel names to keep (e.g., ["TP9", "AF7", "AF8", "TP10"]).
        If None, all channels are kept.
    rename_map : Optional[Dict[str, str]]
        Optional mapping to rename channels before picking (e.g., {"Right AUX": "AUX"}).
    resample_sfreq : float
        Target sampling frequency in Hz (e.g., 256.0). If the file's sfreq already
        matches, resampling is skipped.
    window_s : float
        Window length in seconds (e.g., 1.0).
    stride_s : float
        Stride between consecutive windows in seconds (e.g., 0.25).
    bandpass : Optional[Tuple[float, float]]
        Bandpass in Hz as (low, high). If None, no bandpass is applied.
    notch : Optional[Iterable[float]]
        One or more notch frequencies in Hz (e.g., [50.0] or [60.0]). If None, no notch.
    annotation_label_map : Optional[Dict[str, int]]
        Mapping from annotation description (case-insensitive) to integer label.
        Example for sleep: {"sleep stage w": 0, "sleep stage n2": 2, ...}
        Example for task: {"mental_math": 1, "rest": 0}.
    min_overlap : float
        Minimum fraction of window covered by an annotation to assign its label (0..1).
    verbose : bool
        If True, print MNE logs and additional info.
    allow_unlabeled : bool
        If False, windows without a matching annotation (per min_overlap) are dropped.
        If True, such windows receive label -1 and are returned.
    units_uV : bool
        Convert signal to microvolts (uV). MNE uses Volts internally.
    preload : bool
        Preload data into memory on read.
    stim_channel : Optional[str]
        Name of a stim channel if present; set to None to prevent auto-detection.
    """

    edf_path: str
    pick_channels: Optional[List[str]] = None
    rename_map: Optional[Dict[str, str]] = None
    resample_sfreq: float = 256.0
    window_s: float = 1.0
    stride_s: float = 0.25
    bandpass: Optional[Tuple[float, float]] = None
    notch: Optional[Iterable[float]] = None
    annotation_label_map: Optional[Dict[str, int]] = None
    min_overlap: float = 0.5
    verbose: bool = False
    allow_unlabeled: bool = False
    units_uV: bool = True
    preload: bool = True
    stim_channel: Optional[str] = None


def _import_mne(verbose: bool = False):
    try:
        import mne  # type: ignore
    except Exception as e:  # pragma: no cover - import path
        raise ImportError(
            "mne is required for EDF loading. Install it with 'pip install mne'"
        ) from e
    # Reduce verbosity unless requested
    if not verbose:
        mne.set_log_level("ERROR")
    return mne


def _compute_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return overlap duration (seconds) between intervals [a_start, a_end) and [b_start, b_end)."""
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


def _labels_from_annotations(
    mne_raw,
    window_starts: np.ndarray,
    window_s: float,
    label_map: Dict[str, int],
    min_overlap: float,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Assign a label to each window based on EDF+ annotations.

    Strategy: for each window, compute overlap with each annotation category;
    pick the label with maximum overlap fraction. Assign -1 if below threshold.

    Returns
    -------
    y : np.ndarray, shape (n_windows,)
        Integer labels (or -1 for unlabeled windows).
    counts : Dict[int, int]
        Count of windows per assigned label (excluding -1).
    """
    ann = mne_raw.annotations
    n = window_starts.size
    y = np.full((n,), -1, dtype=int)

    if ann is None or len(ann) == 0:
        return y, {}

    # Normalize label_map keys once
    norm_map = {k.strip().lower(): v for k, v in label_map.items()}

    # Build a list of (onset, end, label)
    labeled_intervals: List[Tuple[float, float, int]] = []
    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        key = str(desc).strip().lower()
        if key in norm_map:
            labeled_intervals.append((float(onset), float(onset + duration), norm_map[key]))

    # Fast path: if no intervals match labels, return all -1
    if not labeled_intervals:
        return y, {}

    counts: Dict[int, int] = {}
    for i, ws in enumerate(window_starts):
        we = ws + window_s
        # Accumulate overlap by label
        best_label = -1
        best_overlap = 0.0
        for a_start, a_end, lab in labeled_intervals:
            overlap = _compute_overlap(a_start, a_end, ws, we)
            if overlap > 0.0 and overlap > best_overlap:
                best_overlap = overlap
                best_label = lab
        if best_label != -1 and best_overlap / window_s >= min_overlap:
            y[i] = best_label
            counts[best_label] = counts.get(best_label, 0) + 1

    return y, counts


def load_edf_windows(
    config: EdfWindowConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """Load an EDF/EDF+ recording and return fixed-length windows.

    Parameters
    ----------
    config : EdfWindowConfig
        Loader configuration.

    Returns
    -------
    X : np.ndarray, shape (n_windows, n_channels, n_times)
        Windowed data (microvolts if config.units_uV is True, otherwise Volts).
    y : Optional[np.ndarray], shape (n_windows,)
        Integer labels per window (or None if no label map provided). Unlabeled windows
        are dropped unless config.allow_unlabeled=True (then labeled as -1).
    meta : Dict
        Metadata including sampling frequency, channel names, window starts (sec), etc.
    """
    mne = _import_mne(verbose=config.verbose)

    # Read EDF
    raw = mne.io.read_raw_edf(
        config.edf_path,
        preload=config.preload,
        stim_channel=config.stim_channel,
        verbose="INFO" if config.verbose else "ERROR",
    )

    # Optional channel rename
    if config.rename_map:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.rename_channels(config.rename_map)

    # Optional pick channels
    if config.pick_channels:
        missing = [c for c in config.pick_channels if c not in raw.ch_names]
        if missing:
            raise ValueError(f"Channels not found in EDF: {missing}. Available: {raw.ch_names}")
        raw.pick(config.pick_channels)

    # Preprocessing (notch, bandpass)
    if config.notch:
        try:
            raw.notch_filter(freqs=list(config.notch), picks="all")
        except Exception as e:
            warnings.warn(f"Notch filter failed: {e}")
    if config.bandpass is not None:
        l_freq, h_freq = config.bandpass
        try:
            raw.filter(l_freq=l_freq, h_freq=h_freq, picks="all")
        except Exception as e:
            warnings.warn(f"Bandpass filter failed: {e}")

    # Resample if needed
    sfreq = float(raw.info["sfreq"])  # original
    if config.resample_sfreq and not math.isclose(sfreq, float(config.resample_sfreq)):
        raw.resample(float(config.resample_sfreq))
    sfreq = float(raw.info["sfreq"])  # new

    # Fixed-length windows via MNE helper (supports overlap)
    # Overlap is window_s - stride_s (cannot be negative)
    overlap = max(0.0, float(config.window_s) - float(config.stride_s))
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=float(config.window_s),
        overlap=overlap,
        preload=True,
        verbose="ERROR",
    )

    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    ch_names = epochs.ch_names
    n_windows_before = X.shape[0]

    # Convert to microvolts if requested
    if config.units_uV:
        X = X * 1e6

    # Compute window start times from epochs.events (samples -> seconds)
    # MNE stores events[:, 0] as the sample index of the event onset.
    window_starts_sec = (epochs.events[:, 0] - raw.first_samp) / sfreq

    y_out: Optional[np.ndarray] = None
    label_counts: Dict[int, int] = {}

    if config.annotation_label_map:
        y_all, label_counts = _labels_from_annotations(
            raw, window_starts_sec, float(config.window_s), config.annotation_label_map, config.min_overlap
        )
        if config.allow_unlabeled:
            y_out = y_all
        else:
            # Drop unlabeled windows (-1)
            keep = y_all != -1
            X = X[keep]
            window_starts_sec = window_starts_sec[keep]
            y_out = y_all[keep]

    meta = {
        "edf_path": config.edf_path,
        "sfreq": sfreq,
        "ch_names": ch_names,
        "window_s": float(config.window_s),
        "stride_s": float(config.stride_s),
        "t_starts": window_starts_sec,
        "n_windows_before_filter": int(n_windows_before),
        "annotation_label_map": dict(config.annotation_label_map) if config.annotation_label_map else None,
        "label_counts": label_counts,
        "units": "uV" if config.units_uV else "V",
    }

    return X, y_out, meta
