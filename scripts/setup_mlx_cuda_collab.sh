#!/usr/bin/env bash
set -euo pipefail

# Colab-friendly MLX (CUDA) setup
# - Installs OpenBLAS/LAPACKE headers to avoid LAPACK_INCLUDE_DIRS-NOTFOUND
# - Forces OpenBLAS vendor to avoid MKL header issues on Colab
# - Supports CUDA arch override via CUDA_ARCHS env var (e.g., 70/75/80/86/89)
# - Builds static MLX and installs into actlib/lib/mlx/install

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLX_SRC="$ROOT_DIR/actlib/lib/mlx"
BUILD_DIR="$MLX_SRC/build"
INSTALL_DIR="$MLX_SRC/install"

echo "[setup_mlx][colab] MLX CUDA build on Colab starting..."

# Verify submodule exists
if ! git -C "$MLX_SRC" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ ! -e "$MLX_SRC/.git" ]; then
    echo "[setup_mlx][colab] ERROR: MLX submodule not found at $MLX_SRC"
    echo "Run: git submodule update --init --recursive"
    exit 1
  fi
fi

# Ensure CMake present
if ! command -v cmake >/dev/null 2>&1; then
  echo "[setup_mlx][colab] Installing cmake..."
  apt-get update -y && apt-get install -y cmake
fi

# Ensure OpenBLAS/LAPACKE headers (avoid MKL header-not-found)
if command -v apt-get >/dev/null 2>&1; then
  echo "[setup_mlx][colab] Installing OpenBLAS/LAPACKE headers..."
  apt-get update -y
  apt-get install -y libopenblas-dev liblapack-dev liblapacke-dev
fi

# Optional: check for nvcc
if ! command -v nvcc >/dev/null 2>&1; then
  echo "[setup_mlx][colab] WARNING: nvcc not found. Ensure CUDA Toolkit is available at /usr/local/cuda."
fi

mkdir -p "$BUILD_DIR" "$INSTALL_DIR"

# Base CMake flags
CMAKE_FLAGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DMLX_BUILD_TESTS=OFF
  -DMLX_BUILD_EXAMPLES=OFF
  -DMLX_BUILD_PYTHON_BINDINGS=OFF
  -DBUILD_SHARED_LIBS=OFF
  -DMLX_BUILD_CUDA=ON
  -DBLA_VENDOR=OpenBLAS
  -DLAPACK_INCLUDE_DIRS=/usr/include
)

# Allow explicit CUDA arch override
if [[ -n "${CUDA_ARCHS:-}" ]]; then
  echo "[setup_mlx][colab] Using CUDA architectures: ${CUDA_ARCHS}"
  CMAKE_FLAGS+=( -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" )
else
  echo "[setup_mlx][colab] Using default CUDA architectures (auto/native). Set CUDA_ARCHS to override."
fi

# Configure, build, install
cmake -S "$MLX_SRC" -B "$BUILD_DIR" "${CMAKE_FLAGS[@]}"

JOBS="8"
if command -v nproc >/dev/null 2>&1; then
  JOBS="$(nproc)"
elif command -v getconf >/dev/null 2>&1; then
  JOBS="$(getconf _NPROCESSORS_ONLN)"
fi
cmake --build "$BUILD_DIR" -j"$JOBS"
cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR"

# Hints
echo ""
echo "[setup_mlx][colab] MLX installed to: $INSTALL_DIR"
echo "[setup_mlx][colab] Set the following when building ACT with MLX:"
echo "  export MLX_INCLUDE=\"$INSTALL_DIR/include\""
echo "  export MLX_LIB=\"$INSTALL_DIR/lib\""
echo "  export MLX_LINK=\"-lmlx\""
echo ""
echo "[setup_mlx][colab] Example ACT build (with cuDNN):"
echo "  make USE_MLX=1 MLX_INCLUDE=\"$INSTALL_DIR/include/\" MLX_LIB=\"$INSTALL_DIR/lib/\" MLX_LINK=\"-lmlx -lcublasLt -lcublas -lcudnn\" -j8"
echo ""
