# CUDA + MLX on Linux: Build and Link Guide

This document explains how to build Apple MLX (C++ core) with CUDA on Linux and how this project links to MLX correctly. It also records the Makefile changes we made to support device linking with `nvcc`, CUDA runtime libraries, and cuBLASLt/cuBLAS.

---

## Overview

- We vendor MLX as a submodule at `third_party/mlx/` and build it with CUDA enabled.
- On Linux with `USE_MLX=1`, we link with `nvcc` (not `g++`) so CUDA device symbols in `libmlx.a` resolve.
- We add CUDA, cuBLASLt, and cuBLAS to the link line and pass rpaths in an `nvcc`-friendly way.
- We forward `-pthread` to the host compiler using `-Xcompiler -pthread` (because `nvcc` rejects raw `-pthread`).
- We keep macOS `-framework` flags off on Linux.

---

## Prerequisites

- NVIDIA GPU + recent driver supporting your CUDA version.
- CUDA Toolkit with `nvcc` installed (e.g., `/usr/local/cuda` or a custom path).
- cuBLAS and cuBLASLt (part of CUDA Toolkit). If your distro uses split dev packages, install them.
- Optional: cuDNN if you need it in your MLX build.
- BLAS/LAPACK (Linux): typically `libblas`/`liblapack` packages.
- Eigen on Linux: `sudo apt-get install -y libeigen3-dev` (we prefer system Eigen via `/usr/include/eigen3`).

---

## Build MLX with CUDA (vendored submodule)

From the project root, we provide a helper script:

```bash
./scripts/setup_mlx_cuda.sh
```

It runs roughly the following steps:

```bash
# Configure
cmake -S third_party/mlx -B third_party/mlx/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DMLX_BUILD_CUDA=ON

# Build and install
cmake --build third_party/mlx/build -j$(nproc)
cmake --install third_party/mlx/build --prefix third_party/mlx/install
```

Artifacts:
- Headers: `third_party/mlx/install/include/`
- Library: `third_party/mlx/install/lib/libmlx.a`

---

## Building this project with MLX CUDA

Use the Makefile flags:

- `USE_MLX=1` — enable MLX integration
- `MLX_INCLUDE` — path to MLX headers (e.g., `third_party/mlx/install/include`)
- `MLX_LIB` — path to MLX libraries (e.g., `third_party/mlx/install/lib`)
- `MLX_LINK` — MLX and optional CUDA deps to link (e.g., `-lmlx`)
- Optional: `CUDA_HOME` — CUDA root if not `/usr/local/cuda`

Example:

```bash
make USE_MLX=1 \
     MLX_INCLUDE="third_party/mlx/install/include" \
     MLX_LIB="third_party/mlx/install/lib" \
     MLX_LINK="-lmlx" -j$(nproc)
```

If your environment needs cuDNN or other CUDA libs, you can extend `MLX_LINK`, e.g.:

```bash
MLX_LINK="-lmlx -lcublasLt -lcublas -lcudnn"
```

Note: Our Makefile already injects `-lcublasLt -lcublas` on Linux when `USE_MLX=1`, so you typically only need `-lmlx` in `MLX_LINK`.

---

## What the Makefile does on Linux with USE_MLX=1

- Switches linker to `nvcc` for executables:
  - `LD := $(CUDA_HOME)/bin/nvcc`
- Ensures link order resolves dependencies:
  1. `$(MLX_LINK)` (e.g., `-lmlx`)
  2. `-lcublasLt -lcublas`
  3. CUDA runtime libs: `-lcudadevrt -lcudart -lcuda -lnvrtc -ldl -lrt -lpthread`
- Adds rpaths in an `nvcc`-compatible way:
  - MLX: `-Xlinker -rpath -Xlinker $(MLX_LIB)`
  - CUDA: `-Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64`
- Forwards pthread to host compiler:
  - `LINK_LDFLAGS := $(filter-out -pthread,$(LDFLAGS)) -Xcompiler -pthread`
- Keeps macOS `-framework` flags guarded under `uname == Darwin` only.

Environment override examples:

```bash
# If CUDA is not at /usr/local/cuda
make USE_MLX=1 CUDA_HOME=/opt/cuda-12.4 ...
```

---

## Runtime library search (rpath vs environment)

We embed rpaths for both `$(MLX_LIB)` and `$(CUDA_HOME)/lib64`. If your environment forbids rpaths or you prefer environment variables, set:

```bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:third_party/mlx/install/lib:$(CUDA_HOME)/lib64"
```

---

## Troubleshooting

- Undefined references like `__cudaRegisterFunction`, `__cudaPopCallConfiguration`, `cudaLaunchKernel`:
  - Cause: linking with `g++` instead of `nvcc` or missing CUDA device runtime.
  - Fix: our Makefile uses `nvcc` as the linker when `USE_MLX=1` (Linux). Ensure `CUDA_HOME` points to a valid toolkit (with `bin/nvcc`).

- `nvcc fatal: Unknown option '-pthread'`:
  - Fix: We forward pthread via `-Xcompiler -pthread` (handled in the Makefile’s `LINK_LDFLAGS`).

- `nvcc fatal: Unknown option '-Wl,-rpath,...'`:
  - Fix: We pass rpaths via `-Xlinker -rpath -Xlinker <path>`.

- `undefined reference to cublasLt*` or `cublas*`:
  - Fix: We link `-lcublasLt -lcublas`. Ensure these libraries exist in `$(CUDA_HOME)/lib64` (or adjust `CUDA_HOME`).

- `cannot find -lcudadevrt`:
  - Ensure full CUDA toolkit is installed. If your toolchain does not ship `libcudadevrt`, you can remove that flag in the Makefile (only if your `nvcc` link does not require it).

- `libmlx.so: cannot open shared object file` (if building shared MLX):
  - Either keep rpaths or export `LD_LIBRARY_PATH` to include the MLX and CUDA lib directories.

---

## Eigen on Linux

We prefer the system’s Eigen headers on Linux and add `/usr/include/eigen3` to the include path. Install with:

```bash
sudo apt-get update && sudo apt-get install -y libeigen3-dev
```

If you previously had a partial `Eigen/` folder in the repo, ensure the compiler uses the system include or switch to `#include <Eigen/Dense>` in sources.

---

## Example targets

- Basic CPU/MLX tests: `bin/test_act`, `bin/test_act_cpu`, `bin/test_act_mlx`, etc.
- WAV CLI using MLX (float32 path): `bin/test_act_mlx_wav`

Example run:

```bash
# Build with MLX CUDA
make USE_MLX=1 \
     MLX_INCLUDE="third_party/mlx/install/include" \
     MLX_LIB="third_party/mlx/install/lib" \
     MLX_LINK="-lmlx" -j$(nproc)

# Run the WAV CLI
./bin/test_act_mlx_wav -i beethoven_16k_mono.wav -p 6 --residual-threshold 1e-6 --float32
```

---

## Notes

- macOS still uses Accelerate and Metal frameworks, but those flags are gated under `uname == Darwin` and are not added on Linux.
- If you change MLX to build shared libraries (`BUILD_SHARED_LIBS=ON`), adjust rpaths or `LD_LIBRARY_PATH` accordingly.
- If your MLX build or usage requires cuDNN or other CUDA libs, extend `MLX_LINK` (e.g., `-lcudnn`).
