#!/usr/bin/env bash
set -euo pipefail

# Ensure we are running under bash (not sourced in zsh)
if [ -z "${BASH_VERSION:-}" ]; then
  echo "[setup_mlx] Please run this script with: bash scripts/setup_mlx.sh"
  # If sourced, return; otherwise exit
  return 1 2>/dev/null || exit 1
fi

# Build and install MLX (C++ library) from the vendored submodule into actlib/lib/mlx/install
# Usage:
#   ./scripts/setup_mlx.sh
# Then build ACT with:
#   make USE_MLX=1 MLX_INCLUDE=$(pwd)/actlib/lib/mlx/install/include MLX_LIB=$(pwd)/actlib/lib/mlx/install/lib MLX_LINK="-lmlx" -j8

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
ROOT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
MLX_SRC="$ROOT_DIR/actlib/lib/mlx"
BUILD_DIR="$MLX_SRC/build"
INSTALL_DIR="$MLX_SRC/install"

# Detect sources (submodule or vendored tree)
if [ ! -d "$MLX_SRC" ] || [ ! -f "$MLX_SRC/CMakeLists.txt" ]; then
  echo "[setup_mlx] ERROR: MLX sources not found at $MLX_SRC"
  echo "Ensure the MLX submodule is initialized or sources are present."
  echo "Run: git submodule update --init --recursive"
  exit 1
fi

# Tooling checks
if ! command -v cmake >/dev/null 2>&1; then
  echo "[setup_mlx] ERROR: cmake not found. Please install cmake (e.g., brew install cmake)"
  exit 1
fi

mkdir -p "$BUILD_DIR" "$INSTALL_DIR"
# Ensure metallib output directory exists for MLX_METAL_PATH
mkdir -p "$INSTALL_DIR/lib"

# Configure MLX
cmake -S "$MLX_SRC" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DMLX_METAL_PATH="$INSTALL_DIR/lib"

# Build MLX (parallel)
JOBS="8"
if command -v sysctl >/dev/null 2>&1; then
  JOBS="$(sysctl -n hw.ncpu)"
fi
cmake --build "$BUILD_DIR" -j"$JOBS"

# Install into actlib/lib/mlx/install
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
echo "  make USE_MLX=1 MLX_INCLUDE=\"$INSTALL_DIR/include\" MLX_LIB=\"$INSTALL_DIR/lib\" MLX_LINK=\"-lmlx\" -j8"
echo ""
echo "[setup_mlx] Note: MLX requires macOS SDK >= 14.0 for Metal, and Xcode command line tools."
