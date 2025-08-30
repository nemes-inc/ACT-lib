import sys
from pathlib import Path
import numpy as np

# Ensure repository python/ is importable
_THIS = Path(__file__).resolve()
PY_ROOT = _THIS.parents[2]  # .../ACT_cpp/python
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

import rbf.rbf_train as rbf_train  # noqa: E402


def test_end_to_end_training_and_prediction():
    np.random.seed(0)

    # 1) Generate synthetic training data using current code
    ds = rbf_train.SyntheticTrainingDataset(n_samples=24)
    X, y = ds.create_labeled_dataset()

    assert X.ndim == 2 and X.shape[1] == 6
    assert y.shape[0] == X.shape[0]

    # Dataset logging
    feature_names = [
        "frequency",
        "chirp_rate",
        "amplitude",
        "duration",
        "time_center",
        "spectral_width",
    ]
    n_pos_total = int((y == 1).sum())
    n_neg_total = int((y == 0).sum())
    print("\n--- Synthetic Dataset ---")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Class distribution -> positives: {n_pos_total}, negatives: {n_neg_total}")
    # Show a few samples from each class
    pos_idx = np.where(y == 1)[0][:3]
    neg_idx = np.where(y == 0)[0][:3]
    def row_to_str(row):
        return ", ".join(f"{name}={val:.3f}" for name, val in zip(feature_names, row))
    if pos_idx.size > 0:
        print("Sample positives:")
        for i in pos_idx:
            print(f"  y=1 idx={i}: {row_to_str(X[i])}")
    if neg_idx.size > 0:
        print("Sample negatives:")
        for i in neg_idx:
            print(f"  y=0 idx={i}: {row_to_str(X[i])}")

    # 2) Train the model (no changes to distance calcs)
    n_centers = max(5, min(10, n_pos_total))  # ensure <= #positives
    clf = rbf_train.RBFClassifier(n_centers=n_centers, kmeans_iters=20, seed=0)
    clf.train_on_synthetic(X, y)

    assert clf.centers is not None and clf.widths is not None and clf.weights is not None
    assert clf.centers.shape[0] == n_centers and clf.centers.shape[1] == 6
    assert clf.widths.shape == (n_centers,)
    assert clf.weights.shape == (n_centers,)

    # Training summary
    print("\n--- Trained RBF Model ---")
    print(f"n_centers: {n_centers}")
    print(
        f"centers shape: {clf.centers.shape}, widths shape: {clf.widths.shape}, weights shape: {clf.weights.shape}"
    )
    print("first 3 centers:")
    for i in range(min(3, clf.centers.shape[0])):
        print(f"  center[{i}]: {row_to_str(clf.centers[i])}")
    print(
        f"widths (min/median/max): {float(np.min(clf.widths)):.4f} / {float(np.median(clf.widths)):.4f} / {float(np.max(clf.widths)):.4f}"
    )
    print(
        f"weights (min/median/max): {float(np.min(clf.weights)):.4f} / {float(np.median(clf.weights)):.4f} / {float(np.max(clf.weights)):.4f}"
    )

    # 3) Basic prediction tests
    gen = rbf_train.SyntheticBetaGammaGenerator()
    fs = gen.fs

    # Few positive and negative trials
    pos_scores = []
    alpha_scores = []
    motor_scores = []

    print(f"\n--- Prediction trials (fs={fs} Hz) ---")
    for t in range(3):
        f = np.random.uniform(18, 35)
        cr = np.random.uniform(0, 5)
        snr = np.random.uniform(-8, 3)
        sig, params = gen.generate_mental_math_burst(
            frequency=f,
            chirp_rate=cr,
            duration=1.0,  # fixed length for speed/caching
            snr_db=snr,
        )
        s = float(clf.predict_realtime(sig, fs=fs))
        pos_scores.append(s)
        print(
            f"pos[{t}]: f={f:.2f}Hz, chirp_rate={cr:.3f}, snr_db={snr:.1f} -> score={s:.4f}"
        )

        sig_alpha = gen.generate_alpha_burst(duration=1.0)
        s_alpha = float(clf.predict_realtime(sig_alpha, fs=fs))
        alpha_scores.append(s_alpha)
        print(f"alpha[{t}]: duration=1.0 -> score={s_alpha:.4f}")

        motor_f = np.random.uniform(15, 25)
        sig_motor = gen.generate_motor_beta(duration=1.0, freq=motor_f)
        s_motor = float(clf.predict_realtime(sig_motor, fs=fs))
        motor_scores.append(s_motor)
        print(f"motor[{t}]: f={motor_f:.2f}Hz, duration=1.0 -> score={s_motor:.4f}")

    # Finite outputs
    assert np.all(np.isfinite(pos_scores))
    assert np.all(np.isfinite(alpha_scores))
    assert np.all(np.isfinite(motor_scores))

    print("\n--- Score arrays ---")
    print(f"pos_scores:   {np.array(pos_scores)}")
    print(f"alpha_scores: {np.array(alpha_scores)}")
    print(f"motor_scores: {np.array(motor_scores)}")

    # Positives should score higher on average than negatives
    pos_med = float(np.median(pos_scores))
    neg_med = float(np.median(alpha_scores + motor_scores))
    assert pos_med > neg_med, f"Expected positives > negatives, got {pos_med:.3f} <= {neg_med:.3f}"
