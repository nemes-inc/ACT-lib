from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import (
    EdfWindowConfig,
    load_edf_windows,
    ActExtractorConfig,
    ActFeatureExtractor,
    MLPConfig,
    build_mlp,
)
from .act_dict_io import read_act_dict_header, summarize_header


def _parse_annotation_map(arg: Optional[str]) -> Optional[Dict[str, int]]:
    if not arg:
        return None
    # Support inline JSON or @path.json
    if arg.startswith("@"):
        p = Path(arg[1:])
        return json.loads(p.read_text())
    return json.loads(arg)


def determine_window_from_dict(dict_path: Path) -> Tuple[float, float, int]:
    """Return (fs, window_s, length) from an existing dictionary file header."""
    h = read_act_dict_header(str(dict_path))
    if h is None:
        raise ValueError(f"Not a valid ACT dictionary: {dict_path}")
    fs = float(h.fs)
    length = int(h.length)
    window_s = float(length) / fs
    return fs, window_s, length


def load_data(
    edf_paths: List[Path],
    resample_sfreq: float,
    window_s: float,
    stride_s: float,
    annotation_map: Optional[Dict[str, int]],
    allow_unlabeled: bool,
    pick_channels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    meta_agg = {"files": []}

    for p in edf_paths:
        cfg = EdfWindowConfig(
            edf_path=str(p),
            pick_channels=pick_channels,
            resample_sfreq=float(resample_sfreq),
            window_s=float(window_s),
            stride_s=float(stride_s),
            annotation_label_map=annotation_map,
            min_overlap=0.5,
            allow_unlabeled=allow_unlabeled,
            verbose=False,
        )
        X, y, meta = load_edf_windows(cfg)
        X_list.append(X)
        if y is not None:
            y_list.append(y)
        meta_agg["files"].append({"path": str(p), "n_windows": int(X.shape[0])})
        # Check consistency of sfreq and length
        if not np.isclose(meta["sfreq"], resample_sfreq):
            raise ValueError(f"File {p} has sfreq {meta['sfreq']} != requested {resample_sfreq}")
        if X.shape[-1] != int(window_s * resample_sfreq):
            raise ValueError("Window length mismatch; check dictionary length vs window_s")

    X_all = np.concatenate(X_list, axis=0)
    if annotation_map is None:
        y_all = np.full((X_all.shape[0],), -1, dtype=int)
    else:
        if not y_list:
            raise ValueError("No labels were extracted from annotations; check annotation_map")
        y_all = np.concatenate(y_list, axis=0)

    return X_all, y_all, meta_agg


def main():
    ap = argparse.ArgumentParser(description="Train MLP on ACT features from EDF files")
    # Data
    ap.add_argument("--edf", nargs="+", help="EDF/EDF+ file(s)")
    ap.add_argument("--pick-channels", nargs="*", default=None, help="Channel names to pick")
    ap.add_argument("--channel-index", type=int, default=0, help="Channel index to extract features from")
    ap.add_argument("--annotation-map", type=str, default=None,
                    help="Inline JSON mapping or @path.json (e.g., '{\"sleep stage n2\":0,\"sleep stage r\":1}')")
    ap.add_argument("--allow-unlabeled", action="store_true", help="Keep unlabeled windows with y=-1")

    # Dictionary selection
    ap.add_argument("--dict-file", type=str, default=None, help="Path to existing dictionary file to load")
    ap.add_argument("--dict-ranges-json", type=str, default=None, help="JSON file with parameter ranges to build a dictionary")
    ap.add_argument("--dict-cache-out", type=str, default=None, help="Where to save a newly built dictionary")
    ap.add_argument("--dict-force", action="store_true", help="Force regenerate the dictionary (overwrite cache)")

    # Signal/window config
    ap.add_argument("--resample", type=float, default=256.0, help="Target sampling frequency (Hz)")
    ap.add_argument("--window-s", type=float, default=None, help="Window length (seconds); defaults to dict length/fs if dict provided, else 1.0")
    ap.add_argument("--stride-s", type=float, default=0.25, help="Stride (seconds)")

    # Training
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--model-out", type=str, default="mlp_model.joblib")
    ap.add_argument("--order", type=int, default=1, help="Transform order (top-k chirplets per window)")

    args = ap.parse_args()

    edf_paths = [Path(p) for p in (args.edf or [])]
    if not edf_paths:
        raise SystemExit("No EDF files provided")

    # Determine dictionary usage
    dict_file = Path(args.dict_file) if args.dict_file else None
    dict_ranges_path = Path(args.dict_ranges_json) if args.dict_ranges_json else None

    if dict_file and dict_ranges_path:
        raise SystemExit("Provide either --dict-file OR --dict-ranges-json, not both")

    # Defaults prior to dict
    resample_sfreq = float(args.resample)
    window_s = float(args.window_s) if args.window_s is not None else 1.0

    extractor_cfg = ActExtractorConfig(
        fs=resample_sfreq,
        length=int(window_s * resample_sfreq),
        dict_cache_file="",
        force_regenerate=False,
        mute=True,
    )

    if dict_file:
        if not dict_file.exists():
            raise SystemExit(f"Dictionary file not found: {dict_file}")
        fs_d, window_s_d, length_d = determine_window_from_dict(dict_file)
        resample_sfreq = fs_d
        window_s = window_s_d
        extractor_cfg.fs = fs_d
        extractor_cfg.length = length_d
        extractor_cfg.dict_cache_file = str(dict_file)
        extractor_cfg.force_regenerate = False
        print(f"Using existing dictionary: {dict_file}")
    elif dict_ranges_path:
        if not args.dict_cache_out:
            raise SystemExit("--dict-cache-out is required when building a new dictionary")
        # Load parameter ranges
        ranges = json.loads(Path(dict_ranges_path).read_text())
        extractor_cfg.ranges = ranges
        extractor_cfg.dict_cache_file = str(Path(args.dict_cache_out))
        extractor_cfg.force_regenerate = True
        # Ensure length/fs consistent with loader
        extractor_cfg.fs = resample_sfreq
        extractor_cfg.length = int(window_s * resample_sfreq)
        print(f"Building new dictionary to: {extractor_cfg.dict_cache_file}")
    else:
        print("No dictionary specified; a small default will be generated in-memory (not recommended for production)")
        extractor_cfg.dict_cache_file = "act_mlp_cache.bin"
        extractor_cfg.force_regenerate = True

    # Load EDF windows
    annotation_map = _parse_annotation_map(args.annotation_map)
    X, y, meta_agg = load_data(
        edf_paths,
        resample_sfreq=resample_sfreq,
        window_s=window_s,
        stride_s=float(args.stride_s),
        annotation_map=annotation_map,
        allow_unlabeled=bool(args.allow_unlabeled),
        pick_channels=args.pick_channels,
    )

    # Extract ACT features
    extractor = ActFeatureExtractor(extractor_cfg)
    feats = extractor.extract_features_batch(X, channel=int(args.channel_index), order=int(args.order))

    # Drop NaN rows from features and align labels
    valid = ~np.isnan(feats).any(axis=1)
    feats = feats[valid]
    y = y[valid]

    # Train/val split
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        feats, y, test_size=float(args.test_size), random_state=int(args.random_state), stratify=y
    )

    model = build_mlp(MLPConfig(random_state=int(args.random_state)))
    model.fit(X_tr, y_tr)

    # Evaluate
    from .model import evaluate
    metrics = evaluate(model, X_te, y_te)
    print("\n=== Evaluation ===")
    for k, v in metrics.items():
        if k == "report":
            print(v)
        else:
            print(f"{k}: {v}")

    # Save model
    import joblib
    # Build feature names according to order
    base = [
        "fc_hz", "chirp_hz_per_s", "amplitude", "duration_s", "time_center_s", "spectral_width_hz"
    ]
    feat_names: list[str] = []
    for i in range(int(args.order)):
        feat_names.extend([f"{b}_{i+1}" for b in base])
    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "dict_file": extractor_cfg.dict_cache_file,
        "fs": resample_sfreq,
        "length": int(window_s * resample_sfreq),
        "channel_index": int(args.channel_index),
        "order": int(args.order),
        "feature_names": feat_names,
    }, str(out_path))
    print(f"Saved model to: {out_path}")


if __name__ == "__main__":
    main()
