# C++ Adaptive Chirplet Transform (ACT) - C++ Implementation

A high-performance, general-purpose C++ implementation of the Adaptive Chirplet Transform for time-frequency analysis of non‑stationary signals. Suitable for audio, radar/sonar, biomedical (including EEG), and other domains. 

## Current Project Status

This section summarizes the current state of the repository as it is now.

- **Core algorithm**: The baseline implementation performs dictionary-based matching pursuit with unit-energy chirplet generation and BFGS refinement. `search_dictionary` is virtual, enabling backend overrides.
- **CPU backends**: `ACT_CPU` (Eigen + BLAS baseline) and `ACT_Accelerate` (Apple Accelerate-optimized) provide fast CPU execution. BLAS is used on Linux; Accelerate on macOS.
- **MLX backend (GPU, float32)**: `ACT_MLX` enables a GPU-accelerated coarse search using Apple MLX when compiled with `USE_MLX=ON`. It pre-packs the dictionary to device and runs `scores = transpose(dict) @ x` and `argmax` on device. Double precision falls back to the CPU path.
- **Python bindings**: `python/act_bindings` exposes `pyact.mpbfgs` with `ActCPUEngine`, `ActMLXEngine`, and a backward-compatible `ActEngine` (CPU by default). `transform(...)` returns a rich dict. Tests are included.
- **MLX build integration**: `scripts/setup_mlx.sh` installs the vendored MLX into `third_party/mlx/install/`. CMake options `USE_MLX`, `MLX_INCLUDE`, `MLX_LIB`, and `MLX_LINK` configure the Python build. `scripts/build_pyact_mlx.sh` builds the wheel with MLX enabled.
- **EEG utilities**: `python/act_mlp` provides feature extraction based on ACT (defaults to `ActMLXEngine`) and a simple MLP training pipeline.
- **Profiling**: `profile_act.cpp` measures end-to-end timings (dictionary, search, transform, SNR).
- **CLI analyzer**: `eeg_act_analyzer` supports interactive exploration of CSV EEG data and ACT parameters.

__What’s next__
- **Batched multi-signal coarse search**: `A^T @ X` for batches of 4–16 signals (CPU via GEMM, MLX via `matmul`).
- **Top‑k per signal**: Efficient selection for batched scores.
- **Batched transform**: Optional refinement (BFGS) per signal with a small thread pool.

__How to try the faster search today__
- Use `ACT_Accelerate` (CPU) or `ACT_MLX` (float32 MLX when enabled):
  - Instantiate `ACT_Accelerate` or `ACT_MLX`, call `generate_chirplet_dictionary()`, then run `search_dictionary(...)` or `transform(...)`.
  - `ACT_MLX` offloads the coarse search to MLX (GPU) when the project is built with `USE_MLX=ON`; otherwise it transparently falls back to the CPU path.

## Overview

The Adaptive Chirplet Transform (ACT) is a powerful signal processing technique that decomposes signals into chirplets - Gaussian-enveloped sinusoids with time-varying frequency. This implementation provides:

- **High Performance**: BLAS/Accelerate-optimized dictionary search; optional MLX GPU coarse search (float32)
- **Flexible Analysis**: Configurable parameter ranges for different signal types
- **Example Applications**: EEG-oriented examples to demonstrate usage
- **Professional Quality**: Production-ready code with comprehensive testing

### Algorithm Summary (Dictionary Search + Optimization)
This implementation uses a two-stage, greedy matching pursuit approach:

1) Coarse dictionary search
   - Build a discrete grid of chirplet parameters (tc, fc, logDt, c).
   - Generate unit-energy chirplet templates and compute correlations against the current residual.
   - Select the best-scoring atom as the initialization.

2) Local continuous optimization
   - Refine the selected atom’s parameters via BFGS over (tc, fc, logDt, c) to maximize correlation.
   - Estimate the optimal coefficient (least-squares against unit-energy template).

3) Greedy update and iterate
   - Subtract the reconstructed chirplet from the residual and repeat steps (1–2) up to the chosen transform order K.

Performance notes: The heavy step is the dictionary search. On CPU it is accelerated by BLAS/Accelerate; when built with `USE_MLX=ON`, the float32 MLX backend offloads the coarse search to the GPU. Unit-energy normalization removes duration bias and stabilizes coefficient estimation.

## Features

### Core Capabilities
- **Adaptive chirplet decomposition** with BFGS refinement (unit-energy atoms)
- **CPU acceleration** using Apple Accelerate (macOS) or BLAS (Linux)
- **Optional MLX acceleration (float32)** for coarse search when built with `USE_MLX=ON`
- **Python bindings** (`pyact.mpbfgs`) with `ActCPUEngine`, `ActMLXEngine`, and `ActEngine` (CPU compatibility wrapper)
- **Configurable dictionary parameters** and on-disk dictionary caching (ACTDICT v2)
- **EEG feature extraction & MLP** utilities in `python/act_mlp`
- **Tests & profiling**: C++ and Python tests, plus a profiling target

## Quick Start

### Prerequisites
- C++17 compatible compiler
- macOS: Xcode Command Line Tools (Accelerate framework)
- Linux: BLAS/LAPACK (`sudo apt-get install libblas-dev liblapack-dev`)
- Python 3.8+ (for bindings), `pip`

### Building (C++ targets)
```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Build all targets
make all

# Run basic test
make test
```

### Python usage (pyact.mpbfgs)
```python
import numpy as np
from pyact.mpbfgs import ActCPUEngine, ActMLXEngine

fs, length = 256.0, 256
ranges = dict(
    tc_min=0, tc_max=length-1, tc_step=8,
    fc_min=2, fc_max=20, fc_step=2,
    logDt_min=-3, logDt_max=-1, logDt_step=0.5,
    c_min=-10, c_max=10, c_step=5,
)

# CPU backend (double)
cpu = ActCPUEngine(fs, length, ranges, True, True, "")

# MLX backend (float32); falls back to CPU if MLX not compiled in
mlx = ActMLXEngine(fs, length, ranges, True, True, "")

x = np.random.randn(length)
out = mlx.transform(x, order=3)
print("Error:", float(out["error"]))
print("First component params:", out["params"][0])

# Backward-compatible wrapper (CPU default)
from pyact.mpbfgs import ActEngine
cpu_compat = ActEngine(fs, length, ranges, False, True, True, "")
```

### Python bindings (CPU-only default)
```bash
# From repo root (optionally in a venv)
python3 -m pip install -v ./python/act_bindings
```

### Python bindings (enable MLX acceleration)
0) Initialize the MLX submodule (only once per clone):
```bash
# If you did not clone with --recurse-submodules
git submodule update --init --recursive

# Alternatively, clone with submodules
# git clone --recurse-submodules <repository-url>

# Verify the submodule exists
test -f third_party/mlx/CMakeLists.txt && echo "MLX submodule present"
```

1) Build/install MLX into the vendored path:
```bash
bash scripts/setup_mlx.sh
```
2) Install the extension with MLX enabled. Either:
```bash
# A) Use environment variable (easiest)
CMAKE_ARGS="-DUSE_MLX=ON \
           -DMLX_INCLUDE=$(pwd)/third_party/mlx/install/include \
           -DMLX_LIB=$(pwd)/third_party/mlx/install/lib \
           -DMLX_LINK=-lmlx" \
 python3 -m pip install -v ./python/act_bindings

# B) Use repeated --config-settings (pip)
python3 -m pip install -v ./python/act_bindings \
 --config-settings=cmake.args=-DUSE_MLX=ON \
 --config-settings=cmake.args=-DMLX_INCLUDE=$(pwd)/third_party/mlx/install/include \
 --config-settings=cmake.args=-DMLX_LIB=$(pwd)/third_party/mlx/install/lib \
 --config-settings=cmake.args=-DMLX_LINK=-lmlx

# Or simply run the convenience script (does A for you)
chmod +x scripts/build_pyact_mlx.sh
./scripts/build_pyact_mlx.sh
```

During configure you should see:
- `pyact: USE_MLX=ON`
- `pyact: MLX_INCLUDE=...`
- `pyact: Found MLX header at .../mlx/mlx.h`

### Running specialized tests
```bash
# MLX backend test (runs CPU if MLX not enabled)
make test_act_mlx

# Performance profiling
make profile

# Python tests
python3 -m pytest -q python/act_bindings/tests
```

### Interactive EEG ACT Analyzer (CLI)
An interactive command-line tool is included to explore EEG CSV data and run ACT over selected windows.

Build the analyzer and run it:
```bash
# Build the analyzer (or `make all`)
make eeg-analyzer

# Run the interactive CLI
./bin/eeg_act_analyzer
```

Example session using the included sample data `data/muse-testdata.csv` (Muse TP9, fs=256 Hz):
```
> load_csv data/muse-testdata.csv
> select 1 0 2048                # column_index start_sample num_samples
> show_params                    # view current tc/fc/logDt/c ranges and size estimate
> params fc 25 49 1              # set frequency range (Hz)
> params logDt -3.0 -0.7 0.3     # set log-duration grid
> params c -15 15 3              # set chirp-rate grid (Hz/s)
> create_dictionary              # builds dictionary for current window length
> analyze 5 0.01                 # find top 5 chirplets, stop if residual < 0.01

# Sliding-window analysis over samples with overlap
> analyze_samples 3 4096 256     # num_chirps end_sample overlap
> exit
```

Available commands in the CLI (`eeg_act_analyzer.cpp`):
- `load_csv <filepath>`: Load a CSV; first row is treated as headers. Non-numeric cells become NaN.
- `select <column_idx> <start_sample> <num_samples>`: Choose a segment. DC offset is removed. NaNs are filtered.
- `params <tc|fc|logDt|c> <min> <max> <step>`: Adjust dictionary parameter ranges.
- `show_params`: Print current parameter ranges and an estimated dictionary memory footprint.
- `create_dictionary`: Construct the dictionary for the selected segment length.
- `analyze <num_chirplets> <residual_threshold>`: Run ACT and print chirplet parameters and residuals.
- `analyze_samples <num_chirps> <end_sample> <overlap>`: Slide a window of dictionary length across the signal.
- `help` / `exit`.

Notes:
- Sampling frequency defaults to 256 Hz (Muse). Adjust code if your data differs.
- `tc` is in samples; reported time is converted to seconds as `tc / fs`.
- Duration is reported as `1000 * exp(logDt)` in milliseconds.
- The analyzer uses `linenoise` for history; a local history file `.eeg_analyzer_history.txt` is created.

## Architecture

### Class Hierarchy
```
ACT (Base)
├── ACT_CPU (Eigen + BLAS baseline)
├── ACT_Accelerate (Apple Accelerate-optimized CPU)
└── ACT_MLX (MLX-accelerated coarse search for float32; CPU fallback otherwise)
```

### Key Components
- **ACT_CPU.h/.cpp**: Eigen + BLAS baseline backend
- **ACT_Accelerate.h/.cpp**: Accelerate-optimized backend (macOS) with BLAS fallback
- **ACT_MLX.h/.cpp**: MLX float32 device dictionary + matmul/argmax fast path
- **python/act_bindings**: pybind11 extension exposing `pyact.mpbfgs`
- **python/act_mlp**: EEG feature extraction and MLP training utilities
- **test_*.cpp**: C++ unit and smoke tests
- **profile_act.cpp**: End-to-end profiling of dictionary, search, and transform
- **scripts/**: `setup_mlx.sh`, `build_pyact_mlx.sh` helpers

## Algorithm Details

### Two-Stage Process
1. **Dictionary Search**: Fast discrete parameter matching
2. **BFGS Optimization**: Continuous parameter refinement

### Parameter Space
- **tc (Time Center)**: When the chirplet occurs
- **fc (Frequency Center)**: Base frequency of oscillation  
- **logDt (Duration)**: Logarithmic duration parameter
- **c (Chirp Rate)**: Frequency modulation rate (Hz/s)

### Chirplet Generation: C++ vs Python Reference

The original Python reference implementation generated chirplets without unit-energy normalization.

```python
def g(self, tc=0, fc=1, logDt=0, c=0):
    """
    tc: in SAMPLES; fc: Hz; logDt: log duration; c: Hz/s
    FS: sampling rate; length: number of samples
    """
    tc /= self.FS
    Dt = np.exp(logDt)
    t = np.arange(self.length)/self.FS
    gaussian_window = np.exp(-0.5 * ((t - tc)/(Dt))**2)
    complex_exp = np.exp(2j*np.pi * (c*(t-tc)**2 + fc*(t-tc)))
    final_chirplet = gaussian_window * complex_exp
    if not self.complex:
        final_chirplet = np.real(final_chirplet)
    if self.float32:
        final_chirplet = final_chirplet.astype(np.float32)
    return final_chirplet
```

In C++ (`ACT.cpp`), we made two critical changes to match principled ACT behavior and fix duration bias:

- Unit-energy normalization: every generated chirplet `g` is L2-normalized to have unit energy. Without normalization, longer-duration atoms systematically win during dictionary search, forcing `logDt` to the upper bound and degrading recovery. Normalization removes this bias and aligns the objective with correlation rather than raw energy.
- Coefficient estimation scaling: with unit-energy atoms, the optimal coefficient is just the dot product between the signal and the chirplet. We removed an incorrect division by the sampling rate (FS) that had been applied previously. With normalization, this yields correct amplitudes and SNR.

Additional notes:
- Real output uses cosine consistently when `complex_mode=false` (real part of the complex exponential), matching the Python `np.real(...)` behavior.
- The same unit-energy normalization is implemented in the SIMD code paths (`ACT_SIMD.cpp`) using Apple Accelerate (vDSP) on macOS and NEON helpers on ARM for efficiency.
- These changes were validated by strict synthetic tests (noiseless and 0 dB noisy), demonstrating accurate parameter recovery and SNR improvement, and eliminating the previous `logDt` upper-bound bias.

### Dictionary Design
The dictionary contains pre-computed chirplet templates for all parameter combinations:
```
Dictionary Size = tc_steps × fc_steps × logDt_steps × c_steps
```

Optimal parameter resolution balances:
- **Temporal Resolution**: Finer steps detect more precise timing
- **Frequency Resolution**: Better frequency discrimination
- **Memory Usage**: Larger dictionaries require more RAM
- **Computation Time**: More templates increase search time

## Usage Examples

### Basic ACT Analysis (Accelerate backend)
```cpp
#include "ACT_Accelerate.h"

// Create parameter ranges
ACT_CPU::ParameterRanges ranges(0, 2047, 8.0,     // time: 0-2047, step 8
                                25.0, 50.0, 2.0,  // freq: 25-50Hz, step 2Hz
                                -3.0, -1.0, 0.5,  // duration: log scale
                                -10.0, 10.0, 5.0);// chirp rate: ±10 Hz/s

// Initialize ACT_Accelerate (uses Accelerate on macOS; BLAS on Linux)
ACT_Accelerate act(256.0, 2048, ranges, /*verbose=*/true);
act.generate_chirplet_dictionary();

// Analyze signal
auto result = act.transform(signal, 5);  // Find top 5 chirplets
```

### Using the MLX backend (float32 GPU coarse search when enabled)
```cpp
#include "ACT_MLX.h"

ACT_CPU::ParameterRanges ranges(/* ... */);
ACT_MLX act(256.0, signal_length, ranges, /*verbose=*/true);
act.generate_chirplet_dictionary();
auto result = act.transform(eeg_signal, 10);
```

Initial findings on performance using the included dataset:

1. **Dictionary Resolution Impact**: Coarse temporal resolution (0.25s steps) caused artificial clustering of chirplets at signal boundaries
2. **Duration Diversity**: Limited logDt values severely constrained duration diversity, leading to uniform chirplet durations
3. **Optimization Benefits**: Two-stage process (dictionary + BFGS) enables detection of precise frequencies (e.g., 28.3 Hz) between discrete dictionary steps
4. **Memory vs. Resolution Trade-off**: Balanced parameter ranges achieve good temporal diversity while maintaining feasible memory usage

## Dependencies

### ALGLIB
This project includes ALGLIB for numerical optimization:
- **Location**: `alglib/` directory
- **Usage**: BFGS optimization of chirplet parameters
- **License**: GPL/Commercial (see alglib/license.txt)
- **Version**: 3.x (included)

### Platform Libraries
- **macOS**: Accelerate framework (automatic)
- **Linux**: BLAS/LAPACK (`libblas-dev liblapack-dev`)
- **Optional MLX (macOS)**: Apple MLX (Metal-based). Built via `scripts/setup_mlx.sh` into `third_party/mlx/install/`.
  - This repository vendors MLX as a git submodule at `third_party/mlx/`. Initialize it with:
    ```bash
    git submodule update --init --recursive
    ```
  - If the submodule is missing, `scripts/setup_mlx.sh` will print an error and suggest this command.

### Python (bindings and tools)
- `numpy`, `pybind11`, `scikit-build-core`
- `pytest` (tests), `pytest-timeout` (optional)

## File Structure

```
ACT_cpp/
├── ACT_CPU.h/.cpp            # Eigen + BLAS baseline backend
├── ACT_Accelerate.h/.cpp     # Accelerate-optimized backend (macOS)
├── ACT_MLX.h/.cpp            # MLX float32 device dictionary + matmul/argmax
├── ACT.h/.cpp (if present)   # Base helpers (historical)
├── alglib/                   # ALGLIB numerical library
├── Eigen/                    # Bundled Eigen headers
├── python/
│   ├── act_bindings/         # pybind11 extension (pyact.mpbfgs)
│   └── act_mlp/              # EEG feature extraction + MLP utilities
├── scripts/                  # setup_mlx.sh, build_pyact_mlx.sh, etc.
├── profile_act.cpp           # Performance profiling
├── data/                     # Sample EEG data
│   └── muse-testdata.csv
├── Makefile                  # Native build system
└── README.md                 # This file
```

### Background

Adapted from the python code at : [https://github.com/amanb2000/Adaptive_Chirplet_Transform](https://github.com/amanb2000/Adaptive_Chirplet_Transform)

### Published Work
Based on the seminal paper:
> Mann, S., & Haykin, S. (1992). The chirplet transform: A generalization of Gabor's logon. *Vision Interface*, 92, 205-212.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `make test` and `python3 -m pytest -q python/act_bindings/tests`
6. Submit a pull request

### Troubleshooting
- **MLX header not found (`mlx/mlx.h`)**: Ensure you ran `scripts/setup_mlx.sh`. Pass MLX paths to CMake via `-DMLX_INCLUDE`, `-DMLX_LIB`, and `-DMLX_LINK=-lmlx`. The Python binding prints `pyact: MLX_INCLUDE=...` during configure.
- **Pip `--config-settings` flags ignored**: Each `-D` must be a separate `--config-settings=cmake.args=...` entry, or use the `CMAKE_ARGS` environment variable.
- **`lipo: can't figure out the architecture type of: .../pyenv/shims/cmake`**: Noisy but harmless if the build proceeds. Prefer a Homebrew CMake ahead of pyenv shims in `PATH` if desired.
- **No MLX speedup observed**: The MLX path accelerates the coarse search for float32 (`ACT_MLX_f`). Double precision falls back to CPU. In Python, `ActMLXEngine` uses float32 internally; ensure you built the wheel with `USE_MLX=ON`.

## Roadmap
- Batched multi‑signal coarse search on CPU (single GEMM: `A^T @ X`) and MLX (`matmul(transpose(dict), X)`).
- Top‑k selection per signal and optional batched refinement.
- Additional benchmarks and CI for MLX-enabled wheels.

### ALGLIB Integration
The implementation includes the complete ALGLIB source tree for:
- Bounded nonlinear optimization (`minbc` family)
- Numerical gradient computation
- OptGuard gradient verification (optional)
- Robust error handling and convergence checking

## License

Same as parent project - see main LICENSE file.
