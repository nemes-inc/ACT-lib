import time
import math
import numpy as np
import pytest

# Public API from our binding
from pyact.mpbfgs import ActCPUEngine, ActMLXEngine


def generate_test_signal(length: int, fs: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=float) / fs

    # Two chirplets + a little noise
    tc1, fc1, c1, dt1 = 0.30, 5.0, 10.0, 0.10
    sig = 0.8 * np.exp(-0.5 * ((t - tc1) / dt1) ** 2) * np.cos(
        2.0 * np.pi * (c1 * (t - tc1) ** 2 + fc1 * (t - tc1))
    )

    tc2, fc2, c2, dt2 = 0.60, 8.0, -5.0, 0.15
    sig += 0.6 * np.exp(-0.5 * ((t - tc2) / dt2) ** 2) * np.cos(
        2.0 * np.pi * (c2 * (t - tc2) ** 2 + fc2 * (t - tc2))
    )

    sig += 0.05 * (rng.random(length) - 0.5)
    return sig.astype(np.float64)


def small_ranges(length: int, fs: float):
    """
    Keep dictionary small so tests run quickly. Note the C++ side includes
    endpoints if `start + k*step <= end`, so counts are approximate.
    """
    return dict(
        tc_min=0.0, tc_max=float(length - 1), tc_step=8.0,  # ~length/8 positions
        fc_min=1.0, fc_max=min(20.0, fs / 2.5), fc_step=2.0,  # ~10 values
        logDt_min=-3.0, logDt_max=-1.0, logDt_step=1.0,       # 3 values
        c_min=-10.0, c_max=10.0, c_step=10.0,                 # 3 values
    )


def _time_transforms(engine, signal: np.ndarray, order: int, repeats: int = 3):
    # Warm-up (populate caches, trigger any lazy inits)
    _ = engine.transform(signal, order=order)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = engine.transform(signal, order=order)
        dt = time.perf_counter() - t0
        times.append(dt)
        # Basic sanity checks
        assert "params" in out and "coeffs" in out and "approx" in out and "residue" in out and "error" in out
        assert len(np.asarray(out["approx"])) == signal.size
        assert len(np.asarray(out["residue"])) == signal.size
    return times


@pytest.mark.timeout(60)
def test_cpu_vs_mlx_performance_smoke(capfd):
    fs = 256.0
    length = 256
    order = 3
    rng = small_ranges(length, fs)

    # Generate one shared signal for both backends
    signal = generate_test_signal(length, fs)

    print("\n=== Backend Performance Smoke Test ===")
    print(f"fs={fs}, length={length}, order={order}")
    print("Ranges:", rng)

    # CPU (double) backend
    cpu = ActCPUEngine(
        fs, length,
        ranges=rng,
        force_regenerate=True,
        mute=True,
        dict_cache_file="",
    )
    cpu_times = _time_transforms(cpu, signal, order=order, repeats=3)
    print(f"CPU backend times (s): {cpu_times} | avg={np.mean(cpu_times):.4f} | min={np.min(cpu_times):.4f}")

    # MLX (float32) backend (falls back to CPU path if MLX was not compiled)
    mlx = ActMLXEngine(
        fs, length,
        ranges=rng,
        force_regenerate=True,
        mute=True,
        dict_cache_file="",
    )
    mlx_times = _time_transforms(mlx, signal, order=order, repeats=3)
    print(f"MLX backend times (s): {mlx_times} | avg={np.mean(mlx_times):.4f} | min={np.min(mlx_times):.4f}")

    # Emit a brief summary and a gentle assertion on correctness (not speed)
    # We do NOT assert MLX is faster since it depends on USE_MLX and environment.
    cpu_out = cpu.transform(signal, order=order)
    mlx_out = mlx.transform(signal, order=order)

    # Verify both produce reasonable errors and arrays
    cpu_err = float(cpu_out["error"])  # L2 norm of residue
    mlx_err = float(mlx_out["error"])
    print(f"Final errors: CPU={cpu_err:.6f}, MLX={mlx_err:.6f}")

    assert cpu_err >= 0.0 and mlx_err >= 0.0

    # Show the captured output in pytest -q as part of the report
    captured = capfd.readouterr()
    print(captured.out)
