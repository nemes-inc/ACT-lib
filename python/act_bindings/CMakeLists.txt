cmake_minimum_required(VERSION 3.20)
project(pyact_mpbfgs LANGUAGES CXX)
set(CMAKE_CXX_COMPILER "g++")

# Find pybind11 provided by the PyPI package (scikit-build-core will make it available)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Extension target
pybind11_add_module(mpbfgs
    src/mpem_module.cpp
    ../../ACT.cpp
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

# Install the extension into the pyact package path inside the wheel
install(TARGETS mpbfgs DESTINATION pyact)
