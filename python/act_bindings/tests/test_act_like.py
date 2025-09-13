import math
import numpy as np
import pytest

from pyact.mpbfgs import ActEngine


def generate_test_signal(length: int, fs: float) -> np.ndarray:
    t = np.arange(length, dtype=float) / fs
    signal = np.zeros(length, dtype=float)

    # First chirplet: tc=0.3s, fc=5Hz, c=+10 Hz/s, dt=0.1
    tc1, fc1, c1, dt1 = 0.3, 5.0, 10.0, 0.1
    signal += 0.8 * np.exp(-0.5 * ((t - tc1) / dt1) ** 2) * np.cos(
        2.0 * np.pi * (c1 * (t - tc1) ** 2 + fc1 * (t - tc1))
    )

    # Second chirplet: tc=0.6s, fc=8Hz, c=-5 Hz/s, dt=0.15
    tc2, fc2, c2, dt2 = 0.6, 8.0, -5.0, 0.15
    signal += 0.6 * np.exp(-0.5 * ((t - tc2) / dt2) ** 2) * np.cos(
        2.0 * np.pi * (c2 * (t - tc2) ** 2 + fc2 * (t - tc2))
    )

    # Add small noise for realism
    rng = np.random.default_rng(0)
    signal += 0.05 * (rng.random(length) - 0.5)
    return signal


def test_basic_transform_like_cpp_example():
    # Parameters matching test_act.cpp (small for speed)
    fs = 64.0
    length = 32

    # Parameter grid matching the C++ test
    test_ranges = dict(
        tc_min=0, tc_max=31, tc_step=8,     # 0..32 step 8 (approx)
        fc_min=2, fc_max=12, fc_step=2,     # 2..12 by 2
        logDt_min=-3, logDt_max=-1, logDt_step=1,  # -3,-2,-1
        c_min=-10, c_max=10, c_step=10      # -10, 0, 10
    )

    # Print banner similar to the C++ test
    print("=== Python ACT Binding Test ===\n")

    print("1. Initializing ACT with small test dictionary...")
    # Avoid cache I/O in tests by setting empty cache file and force regenerate
    engine = ActEngine(
        fs, length,
        ranges=test_ranges,
        complex_mode=False,
        force_regenerate=True,
        mute=True,
        dict_cache_file="",
    )
    print("   Engine initialized (fs=", fs, ", length=", length, ")", sep="")

    signal = generate_test_signal(length, fs)
    print("\n2. Generating test signal...")
    np.set_printoptions(precision=4, suppress=True)
    print("   Test signal (first 10 samples):", np.array2string(signal[:10], separator=", "))

    print("\n3. Performing 3-order ACT transform...")
    # Perform 3-order transform (binding now returns full result dict)
    out = engine.transform(signal, order=3)

    # Unpack full results
    params = out["params"]
    coeffs = out["coeffs"]
    error = float(out["error"])
    approx = np.asarray(out["approx"], dtype=float)
    residue = np.asarray(out["residue"], dtype=float)

    print("\n4. Transform Results:")
    print(f"   Final error: {error:.6f}")
    print("   Chirplet parameters:")
    for i, (p, a) in enumerate(zip(params, coeffs), start=1):
        tc, fc, logDt, c = p
        print(
            f"     Chirplet {i}: tc={tc:.2f}, fc={fc:.2f}, logDt={logDt:.2f}, c={c:.2f}, coeff={a:.4f}"
        )

    # Print short previews like the C++ helper
    def preview(vec, name, n=10):
        v = np.asarray(vec, dtype=float)
        head = np.array2string(v[:n], precision=4, separator=", ")
        if v.size > n:
            head = head[:-1] + ", ...]"
        print(f"{name} (size={v.size}): {head}")

    preview(approx, "Approximation")
    preview(residue, "Final residue")

    # Basic correctness checks similar to previous version
    # Validate that arrays have consistent lengths
    assert len(params) == len(coeffs) >= 1
    assert len(approx) == length and len(residue) == length

    # Compute quality metric like C++ example
    signal_energy = float(np.dot(signal, signal))
    residue_energy = float(np.dot(residue, residue))
    assert residue_energy >= 0.0
    if residue_energy > 0:
        snr_db = 10.0 * math.log10(signal_energy / residue_energy)
        print("\n5. Quality Metrics:")
        print(f"   Signal energy: {signal_energy:.4f}")
        print(f"   Residue energy: {residue_energy:.6f}")
        print(f"   SNR: {snr_db:.2f} dB")

    print("\n=== Test completed successfully! ===")
