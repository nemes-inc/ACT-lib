cmake_minimum_required(VERSION 3.20)
project(pyact_mpbfgs LANGUAGES CXX)

# Find pybind11 provided by the PyPI package (scikit-build-core will make it available)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# New: optional MLX acceleration toggle for Python wheels
option(USE_MLX "Enable MLX acceleration in mpbfgs backend" OFF)
set(MLX_INCLUDE "" CACHE PATH "Path to MLX include directory containing mlx/mlx.h")
set(MLX_LIB "" CACHE PATH "Path to MLX library directory")
set(MLX_LINK "" CACHE STRING "Extra MLX link flags, e.g., -lmlx")

# Diagnostics to confirm values during configuration
message(STATUS "pyact: USE_MLX=${USE_MLX}")
message(STATUS "pyact: MLX_INCLUDE=${MLX_INCLUDE}")
message(STATUS "pyact: MLX_LIB=${MLX_LIB}")
message(STATUS "pyact: MLX_LINK=${MLX_LINK}")

# Extension target
pybind11_add_module(mpbfgs
    src/mpem_module.cpp
    ../../ACT_CPU.cpp
    ../../ACT_Accelerate.cpp
    ../../ACT_MLX.cpp
    # ALGLIB sources (generic, portable; exclude x86-specific kernels)
    ../../alglib/alglib-cpp/src/alglibinternal.cpp
    ../../alglib/alglib-cpp/src/alglibmisc.cpp
    ../../alglib/alglib-cpp/src/ap.cpp
    ../../alglib/alglib-cpp/src/dataanalysis.cpp
    ../../alglib/alglib-cpp/src/diffequations.cpp
    ../../alglib/alglib-cpp/src/fasttransforms.cpp
    ../../alglib/alglib-cpp/src/integration.cpp
    ../../alglib/alglib-cpp/src/interpolation.cpp
    ../../alglib/alglib-cpp/src/linalg.cpp
    ../../alglib/alglib-cpp/src/optimization.cpp
    ../../alglib/alglib-cpp/src/solvers.cpp
    ../../alglib/alglib-cpp/src/specialfunctions.cpp
    ../../alglib/alglib-cpp/src/statistics.cpp
)

# Include directories
# - Root for ACT.h
# - ALGLIB headers
# - Current source dir for binding sources

target_include_directories(mpbfgs PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/../../alglib/alglib-cpp/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# C++ standard and warnings
set_target_properties(mpbfgs PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_compile_options(mpbfgs PRIVATE -O3 -Wall -Wextra -fno-fast-math)
target_link_libraries(mpbfgs PRIVATE Python3::NumPy)

if(APPLE)
  # Optimize for Apple Silicon; don't force architecture (scikit-build-core sets it)
    target_compile_definitions(mpbfgs PRIVATE APPLE)
  target_compile_options(mpbfgs PRIVATE -march=native)
  target_link_libraries(mpbfgs PRIVATE "-framework Accelerate")
endif()

# Conditionally enable MLX GPU acceleration
if(USE_MLX)
  # Early validation for MLX headers
  if(NOT MLX_INCLUDE)
    message(FATAL_ERROR "USE_MLX=ON but MLX_INCLUDE is empty. Pass -DMLX_INCLUDE=/path/to/mlx/install/include")
  endif()
  if(NOT EXISTS "${MLX_INCLUDE}/mlx/mlx.h")
    message(FATAL_ERROR "USE_MLX=ON but MLX header not found at ${MLX_INCLUDE}/mlx/mlx.h (MLX_INCLUDE=${MLX_INCLUDE})")
  endif()
  message(STATUS "pyact: Found MLX header at ${MLX_INCLUDE}/mlx/mlx.h")
  target_compile_definitions(mpbfgs PRIVATE USE_MLX)
  if(MLX_INCLUDE)
    target_include_directories(mpbfgs PRIVATE ${MLX_INCLUDE})
  endif()
  if(MLX_LIB)
    target_link_directories(mpbfgs PRIVATE ${MLX_LIB})
  endif()
  if(APPLE)
    target_link_libraries(mpbfgs PRIVATE
      "-framework Metal"
      "-framework MetalPerformanceShaders"
      "-framework Foundation"
      "-framework QuartzCore"
    )
  endif()
  if(MLX_LINK)
    target_link_libraries(mpbfgs PRIVATE ${MLX_LINK})
  endif()
endif()

# Install the extension into the pyact package path inside the wheel
install(TARGETS mpbfgs DESTINATION pyact)
