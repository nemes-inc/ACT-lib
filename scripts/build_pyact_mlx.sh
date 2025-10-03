#!/usr/bin/env bash
set -euo pipefail

# Build and install the pyact Python extension with MLX acceleration enabled.
# This script will:
#  - Ensure the vendored MLX is built and installed into third_party/mlx/install
#  - Set CMAKE_ARGS to enable USE_MLX and point to the installed MLX include/lib
#  - Invoke pip to build and install the wheel for pyact
#
# Usage:
#   ./scripts/build_pyact_mlx.sh
#
# Optional: activate your venv first (recommended)
#   python -m venv python/.venv && source python/.venv/bin/activate

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLX_SRC="$ROOT_DIR/actlib/lib/mlx"
INSTALL_DIR="$MLX_SRC/install"
INCLUDE_DIR="$INSTALL_DIR/include"
LIB_DIR="$INSTALL_DIR/lib"

# Step 1: Ensure MLX is built and installed (header must exist)
if [[ ! -f "$INCLUDE_DIR/mlx/mlx.h" ]]; then
  echo "[build_pyact_mlx] MLX headers not found at $INCLUDE_DIR/mlx/mlx.h"
  echo "[build_pyact_mlx] Building MLX via scripts/setup_mlx.sh ..."
  bash "$ROOT_DIR/scripts/setup_mlx.sh"
else
  echo "[build_pyact_mlx] Found MLX header: $INCLUDE_DIR/mlx/mlx.h"
fi

# Verify header again
if [[ ! -f "$INCLUDE_DIR/mlx/mlx.h" ]]; then
  echo "[build_pyact_mlx] ERROR: MLX header still missing after setup: $INCLUDE_DIR/mlx/mlx.h"
  echo "[build_pyact_mlx] Make sure the MLX submodule is initialized and can be built."
  exit 1
fi

# Step 2: Export CMake arguments for scikit-build-core
export CMAKE_ARGS="-DUSE_MLX=ON -DMLX_INCLUDE=$INCLUDE_DIR -DMLX_LIB=$LIB_DIR -DMLX_LINK=-lmlx"

echo "[build_pyact_mlx] CMAKE_ARGS=$CMAKE_ARGS"

# Step 3: Build and install pyact
cd "$ROOT_DIR"
python3 -m pip install -v ./python/act_bindings

echo "[build_pyact_mlx] Done. Verify MLX was picked up by checking CMake configure output lines:"
echo "  - pyact: USE_MLX=ON"
echo "  - pyact: MLX_INCLUDE=$INCLUDE_DIR"
echo "  - pyact: Found MLX header at $INCLUDE_DIR/mlx/mlx.h"
