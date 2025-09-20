#!/usr/bin/env bash
set -euo pipefail

# Build and install MLX (C++ library) from the vendored submodule into third_party/mlx/install
# Usage:
#   ./scripts/setup_mlx.sh
# Then build ACT with:
#   make USE_MLX=1 MLX_INCLUDE=$(pwd)/third_party/mlx/install/include MLX_LIB=$(pwd)/third_party/mlx/install/lib MLX_LINK="-lmlx" -j8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLX_SRC="$ROOT_DIR/third_party/mlx"
BUILD_DIR="$MLX_SRC/build"
INSTALL_DIR="$MLX_SRC/install"

# Detect submodule: note .git can be a file (gitdir) in submodules
if ! git -C "$MLX_SRC" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ ! -e "$MLX_SRC/.git" ]; then
    echo "[setup_mlx] ERROR: MLX submodule not found at $MLX_SRC"
    echo "Run: git submodule update --init --recursive"
    exit 1
  fi
fi

# Tooling checks
if ! command -v cmake >/dev/null 2>&1; then
  echo "[setup_mlx] ERROR: cmake not found. Please install cmake (e.g., brew install cmake)"
  exit 1
fi

mkdir -p "$BUILD_DIR" "$INSTALL_DIR"

# Configure MLX
cmake -S "$MLX_SRC" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DMLX_BUILD_CUDA=ON

# Build MLX (parallel)
# Detect parallel jobs (Linux-friendly)
JOBS="8"
if command -v nproc >/dev/null 2>&1; then
  JOBS="$(nproc)"
elif command -v getconf >/dev/null 2>&1; then
  JOBS="$(getconf _NPROCESSORS_ONLN)"
else
  JOBS="$(grep -c ^processor /proc/cpuinfo || echo 8)"
fi
cmake --build "$BUILD_DIR" -j"$JOBS"

# Install into third_party/mlx/install
cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR"

# Print environment hints
echo ""
echo "[setup_mlx] MLX installed to: $INSTALL_DIR"
echo "[setup_mlx] Set the following when building ACT with MLX:"
echo "  export MLX_INCLUDE=\"$INSTALL_DIR/include\""
echo "  export MLX_LIB=\"$INSTALL_DIR/lib\""
echo "  export MLX_LINK=\"-lmlx\""
echo ""
echo "[setup_mlx] Example build command:"
echo "  make USE_MLX=1 MLX_INCLUDE=\"$INSTALL_DIR/include/\" MLX_LIB=\"$INSTALL_DIR/lib/\" MLX_LINK=\"-lmlx -lcublasLt -lcublas -lcudnn\" -j8"
echo ""
