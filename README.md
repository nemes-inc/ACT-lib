# C++ Adaptive Chirplet Transform (ACT) - C++ Implementation

A high-performance, general-purpose C++ implementation of the Adaptive Chirplet Transform for time-frequency analysis of non‑stationary signals. Suitable for audio, radar/sonar, biomedical (including EEG), and other domains. 

## Current Project Status

This section summarizes the current state of the repository as it is now.

- __Core algorithm__: The base `ACT` implementation in `ACT.cpp/h` performs dictionary-based matching pursuit with unit-energy chirplet generation and BFGS refinement. `search_dictionary` is now virtual, enabling backend overrides.
- __SIMD backend__: `ACT_SIMD` overrides `search_dictionary` and key primitives using Apple Accelerate (vDSP) and NEON helpers. It remains the primary CPU-optimized path for macOS/ARM.
- __MLX backend scaffold__: `ACT_MLX` is introduced as a drop-in subclass of `ACT`.
  - Today, it provides a fast CPU baseline by flattening the dictionary to a row‑major matrix and performing a single BLAS GEMV (`cblas_dgemv`) to compute all dot products at once. This has shown substantial reductions in dictionary search time and end-to-end transform time in profiling.
  - It includes toggles `enable_mlx(bool)` and `use_precomputed_gemv(bool)`; with MLX disabled (default), it uses the GEMV path. MLX GPU acceleration hooks are scaffolded behind `ACT_USE_MLX` but not yet implemented.
- __Build system__: The `Makefile` includes an optional `USE_MLX` flag and paths for future MLX C++ integration. A new test target `test_act_mlx` demonstrates the `ACT_MLX` backend and runs without requiring MLX (uses the GEMV CPU path by default).
- __Profiling__: `profile_act.cpp` can instantiate `ACT_MLX` and measure end-to-end timings (dictionary search, full transform, SNR). Logs show when the GEMV path is used for search.
- __CLI analyzer__: `eeg_act_analyzer` remains available for interactive exploration of CSV EEG data and ACT parameters.
- __Web UI (p5.js)__: A simple in-browser visual workbench exists under `p5js/` (renamed display title: “ACT Analysis Workbench”).

__What is not yet done__
- MLX GPU execution (precomputed dictionary GEMV on device or tiled on-the-fly chirplet generation with on-device matvec) is not yet implemented. The C++ scaffolding and build flags are in place to add this next.
- The base `ACT` class uses per-atom inner products for dictionary search. The GEMV acceleration is currently provided in `ACT_MLX`.

__How to try the faster search today__
- Use the `ACT_MLX` backend (CPU GEMV path by default):
  - Instantiate `ACT_MLX`, call `generate_chirplet_dictionary()`, then run `search_dictionary(...)` or `transform(...)`.
  - At runtime you can ensure GEMV is used by leaving MLX disabled (`enable_mlx(false)`) and keeping precomputed GEMV on (`use_precomputed_gemv(true)`, default).
  - On macOS, the path uses the Accelerate framework; on Linux, ensure BLAS is available.

## Overview

The Adaptive Chirplet Transform (ACT) is a powerful signal processing technique that decomposes signals into chirplets - Gaussian-enveloped sinusoids with time-varying frequency. This implementation provides:

- **High Performance**: SIMD-optimized dictionary search with multi-threading support
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

Performance notes: The heavy step is the dictionary search, which is accelerated with SIMD (vDSP/NEON) and optional multi-threading across signals. Unit-energy normalization removes duration bias and stabilizes coefficient estimation.

## Features

### Core Capabilities
- Adaptive chirplet decomposition with BFGS optimization
- SIMD acceleration using Apple Accelerate framework
- Multi-threaded processing for large datasets
- Configurable dictionary parameters
- CSV output for analysis results

## Quick Start

### Prerequisites
- C++17 compatible compiler (g++ recommended)
- macOS: Xcode Command Line Tools (for Accelerate framework)
- Linux: BLAS and LAPACK libraries (`sudo apt-get install libblas-dev liblapack-dev`)

### Building
```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Build all targets
make all

# Run basic test
make test
```

### Running specialized tests
```bash
# SIMD performance test
make simd

# Performance profiling
make profile
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
ACT (Base Class)
├── ACT_SIMD (SIMD Optimized)
├── ACT_SIMD_MultiThreaded (SIMD + Multi-threading)
└── ACT_multithreaded (Multi-threading)
```

### Key Components
- **ACT.cpp/h**: Core ACT implementation with BFGS optimization
- **ACT_SIMD.cpp/h**: SIMD-accelerated version using Accelerate framework
- **ACT_SIMD_MultiThreaded.cpp/h**: Combined SIMD and multi-threading
- **ACT_Benchmark.cpp/h**: Performance benchmarking utilities

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

### Basic ACT Analysis
```cpp
#include "ACT_SIMD.h"

// Create parameter ranges
ACT::ParameterRanges ranges(0, 2047, 8.0,     // time: 0-2047, step 8
                           25.0, 50.0, 2.0,   // freq: 25-50Hz, step 2Hz
                           -3.0, -1.0, 0.5,   // duration: log scale
                           -10.0, 10.0, 5.0); // chirp rate: ±10 Hz/s

// Initialize ACT with SIMD optimization
ACT_SIMD act(256.0, 2048, ranges);
act.create_dictionary();

// Analyze signal
auto result = act.transform(signal, 5);  // Find top 5 chirplets
```

### SIMD Optimized ACT
```cpp
ACT_SIMD act(256.0, signal_length, ranges);
act.create_dictionary();

// Perform analysis
auto result = act.transform(eeg_signal, 10);

// Access results
for (size_t i = 0; i < result.params.size(); ++i) {
    double time_center = result.params[i][0] / 256.0;  // seconds
    double frequency = result.params[i][1];            // Hz
    double duration = exp(result.params[i][2]) * 1000; // ms
    double chirp_rate = result.params[i][3];           // Hz/s
    double coefficient = result.coeffs[i];
}
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

## File Structure

```
Adaptive_Chirplet_Transform_Cpp/
├── ACT.cpp/h                 # Core ACT implementation
├── ACT_SIMD.cpp/h           # SIMD optimized version
├── ACT_SIMD_MultiThreaded.* # SIMD + multi-threading
├── ACT_multithreaded.*      # Multi-threading support
├── ACT_Benchmark.*          # Performance utilities
├── test_act.cpp             # Basic functionality test
├── test_eeg_gamma_8s.cpp    # 8-second EEG analysis
├── test_eeg_gamma_30s.cpp   # 30-second EEG analysis
├── test_simd*.cpp           # SIMD performance tests
├── profile_act.cpp          # Performance profiling
├── data/                    # Sample EEG data
│   └── muse-testdata.csv   # Test data collected with Muse headband
├── alglib/                  # ALGLIB numerical library
├── Makefile                 # Build system
├── .gitignore              # Git ignore rules
└── README.md               # This file
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
5. Ensure all tests pass: `make test`
6. Submit a pull request
- **Standard library** (math, algorithm, vector, etc.)
- **Optional**: Valgrind for memory checking

### ALGLIB Integration
The implementation includes the complete ALGLIB source tree for:
- Bounded nonlinear optimization (`minbc` family)
- Numerical gradient computation
- OptGuard gradient verification (optional)
- Robust error handling and convergence checking

## License

Same as parent project - see main LICENSE file.
