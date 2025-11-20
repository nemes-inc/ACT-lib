# Adaptive Chirplet Transform (ACT) - C++ Implementation
# Professional Makefile for Public Release
# 
# Usage:
#   make all          - Build all targets
#   make test         - Run basic ACT test
#   make profile      - Run performance profiling
#   make clean        - Clean build artifacts
#   make help         - Show available targets

# Compiler Configuration
CXX = g++
CC = gcc
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Wuninitialized -g -MMD -MP 
CFLAGS = -O3 -Wall -Wextra -g -MMD -MP 
LDFLAGS = -lm -pthread

INCLUDES_CORE = -Iactlib/include -Iactlib/lib 
INCLUDES_APPS = -Iapps/eeg_act_analyzer/src -Iapps/eeg_act_analyzer/lib -Iapps/eeg_act_analyzer/lib/spdlog/include -Iapps/eeg_act_analyzer/lib/oscpp/include
# Platform-specific optimizations
ifeq ($(shell uname),Darwin)
    CXXFLAGS += -march=native
    LDFLAGS += -framework Accelerate
    # Use updated Accelerate CBLAS/LAPACK headers to avoid deprecation warnings
    CXXFLAGS += -DACCELERATE_NEW_LAPACK
    # Optional: build with ILP64 (64-bit BLAS/LAPACK integer sizes) by passing ILP64=1 to make
    ILP64 ?= 0
    ifeq ($(ILP64),1)
        CXXFLAGS += -DACCELERATE_LAPACK_ILP64
    endif
else ifeq ($(shell uname),Linux)
    CXXFLAGS += -march=native
    LDFLAGS += -lblas -llapack
endif

# Directories
OBJDIR = obj
BINDIR = bin

# Core ACT Sources 
ACT_CORE_SOURCES = actlib/src/ACT.cpp actlib/src/ACT_CPU.cpp actlib/src/ACT_Accelerate.cpp actlib/src/ACT_MLX.cpp

# Test Sources
TEST_ACT_SOURCES = actlib/test/test_act.cpp 
TEST_ACT_CPU_SOURCES = actlib/test/test_act_cpu.cpp
TEST_ACT_CPU_F_SOURCES = actlib/test/test_act_cpu_f.cpp
TEST_ACT_ACCEL_SOURCES = actlib/test/test_act_accel.cpp
TEST_ACT_CPU_MT_SOURCES = actlib/test/test_act_cpu_mt.cpp
TEST_ACT_SYNTHETIC_SOURCES = actlib/test/test_act_synthetic.cpp
TEST_ACT_CHALLENGING_SOURCES = actlib/test/test_act_challenging.cpp
PROFILE_ACT_SOURCES = actlib/test/profile_act.cpp
PROFILE_ACT_MT_SOURCES = actlib/test/profile_act_mt.cpp

# Linenoise library
LINENOISE_SOURCES = apps/eeg_act_analyzer/lib/linenoise/linenoise.c
LINENOISE_OBJECTS = $(OBJDIR)/apps/eeg_act_analyzer/lib/linenoise/linenoise.o

# App Sources
EEG_ACT_ANALYZER_SOURCES = apps/eeg_act_analyzer/src/eeg_act_analyzer.cpp apps/eeg_act_analyzer/src/muse_osc_receiver.cpp
EEG_ACT_ANALYZER_OBJECTS = $(OBJDIR)/apps/eeg_act_analyzer/src/eeg_act_analyzer.o $(OBJDIR)/apps/eeg_act_analyzer/src/muse_osc_receiver.o

# ALGLIB Dependencies (Essential Components)
ALGLIB_SOURCES = \
	actlib/lib/alglib/alglib-cpp/src/ap.cpp \
	actlib/lib/alglib/alglib-cpp/src/alglibinternal.cpp \
	actlib/lib/alglib/alglib-cpp/src/alglibmisc.cpp \
	actlib/lib/alglib/alglib-cpp/src/optimization.cpp \
	actlib/lib/alglib/alglib-cpp/src/linalg.cpp \
	actlib/lib/alglib/alglib-cpp/src/solvers.cpp \
	actlib/lib/alglib/alglib-cpp/src/dataanalysis.cpp \
	actlib/lib/alglib/alglib-cpp/src/interpolation.cpp \
	actlib/lib/alglib/alglib-cpp/src/specialfunctions.cpp \
	actlib/lib/alglib/alglib-cpp/src/statistics.cpp \
	actlib/lib/alglib/alglib-cpp/src/fasttransforms.cpp \
	actlib/lib/alglib/alglib-cpp/src/integration.cpp \
	actlib/lib/alglib/alglib-cpp/src/diffequations.cpp

# Object Files
ACT_CORE_OBJECTS = $(ACT_CORE_SOURCES:%.cpp=$(OBJDIR)/%.o)
ALGLIB_OBJECTS = $(ALGLIB_SOURCES:%.cpp=$(OBJDIR)/%.o)

# Optional: MLX integration
# Enable with: make USE_MLX=1 MLX_INCLUDE=/path/to/mlx/include MLX_LIB=/path/to/mlx/lib
USE_MLX ?= 0
MLX_INCLUDE ?=
MLX_LIB ?=
MLX_LINK ?=
ifneq ($(MLX_INCLUDE),)
    CXXFLAGS += -I$(MLX_INCLUDE)
endif
ifeq ($(USE_MLX),1)
    CXXFLAGS += -DUSE_MLX
	# Apple GPU frameworks often used with MLX
	ifeq ($(shell uname),Darwin)
		LDFLAGS += -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework QuartzCore
		ifneq ($(MLX_LIB),)
			LDFLAGS += -L$(MLX_LIB)
		endif
		ifneq ($(MLX_LINK),)
			LDFLAGS += $(MLX_LINK)
		endif
    else ifeq ($(shell uname),Linux)
        # Link CUDA runtime for MLX CUDA backend on Linux
        CUDA_HOME ?= /usr/local/cuda
        NVCC ?= $(CUDA_HOME)/bin/nvcc
        # Use nvcc as the linker to resolve CUDA registration symbols from MLX static libs
        LD := $(NVCC)
        ifneq ($(MLX_LIB),)
            LDFLAGS += -L$(MLX_LIB) -Xlinker -rpath -Xlinker $(MLX_LIB)
        endif
        ifneq ($(MLX_LINK),)
            # Place MLX before CUDA libs so its dependencies resolve correctly
            LDFLAGS += $(MLX_LINK)
        endif
        # MLX CUDA backend depends on cuBLASLt/cuBLAS for GEMM kernels
        LDFLAGS += -lcublasLt -lcublas
        # Now add CUDA libs after MLX
        LDFLAGS += -L$(CUDA_HOME)/lib64 -Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64 -lcudadevrt -lcudart -lcuda -lnvrtc -ldl -lrt -lpthread
        # Place LAPACK/BLAS after MLX & CUDA so LAPACK symbols from libmlx.a resolve
        LDFLAGS += -llapack -lblas
        # nvcc does not accept '-pthread'; pass it to the host compiler instead
        LINK_LDFLAGS := $(filter-out -pthread,$(LDFLAGS)) -Xcompiler -pthread
    endif
endif

# Executables
TEST_ACT_TARGET = $(BINDIR)/test_act
TEST_ACT_CPU_TARGET = $(BINDIR)/test_act_cpu
TEST_ACT_CPU_F_TARGET = $(BINDIR)/test_act_cpu_f
TEST_ACT_ACCEL_TARGET = $(BINDIR)/test_act_accel
TEST_ACT_CPU_MT_TARGET = $(BINDIR)/test_act_cpu_mt
PROFILE_ACT_TARGET = $(BINDIR)/profile_act
TEST_ACT_SYNTHETIC_TARGET = $(BINDIR)/test_act_synthetic
TEST_ACT_CHALLENGING_TARGET = $(BINDIR)/test_act_challenging
TEST_ACT_MLX_TARGET = $(BINDIR)/test_act_mlx
PROFILE_ACT_MT_TARGET = $(BINDIR)/profile_act_mt
EEG_ACT_ANALYZER_TARGET = $(BINDIR)/eeg_act_analyzer
TEST_DICT_IO_TARGET = $(BINDIR)/test_dict_io

# Default target
all: $(TEST_ACT_TARGET) $(TEST_ACT_CPU_TARGET) $(TEST_ACT_CPU_F_TARGET) $(TEST_ACT_ACCEL_TARGET) $(TEST_ACT_CPU_MT_TARGET) $(PROFILE_ACT_TARGET) $(PROFILE_ACT_MT_TARGET) $(TEST_ACT_SYNTHETIC_TARGET) $(TEST_ACT_CHALLENGING_TARGET) $(EEG_ACT_ANALYZER_TARGET) $(TEST_DICT_IO_TARGET) $(TEST_ACT_MLX_TARGET)

# Create directories
$(OBJDIR):
	@mkdir -p $(OBJDIR)/actlib/src
	@mkdir -p $(OBJDIR)/actlib/lib/alglib/alglib-cpp/src
	@mkdir -p $(OBJDIR)/actlib/test
	@mkdir -p $(OBJDIR)/actlib/test/lib
	@mkdir -p $(OBJDIR)/apps/src
	@mkdir -p $(OBJDIR)/apps/eeg_act_analyzer/src
	@mkdir -p $(OBJDIR)/apps/eeg_act_analyzer/lib/linenoise

$(BINDIR):
	@mkdir -p $(BINDIR)

# Object file compilation rules
# More specific patterns must come first to avoid being overridden by generic rules

$(OBJDIR)/actlib/lib/alglib/alglib-cpp/src/%.o: actlib/lib/alglib/alglib-cpp/src/%.cpp | $(OBJDIR)
	@echo "Compiling ALGLIB $<..."
	@$(CXX) $(CXXFLAGS) $(INCLUDES_CORE) -c $< -o $@

$(OBJDIR)/apps/eeg_act_analyzer/lib/linenoise/linenoise.o: apps/eeg_act_analyzer/lib/linenoise/linenoise.c | $(OBJDIR)
	@echo "Compiling LINENOISE $<..."
	@$(CC) $(CFLAGS) $(INCLUDES_APPS) -c $< -o $@

$(OBJDIR)/apps/eeg_act_analyzer/src/%.o: apps/eeg_act_analyzer/src/%.cpp | $(OBJDIR)
	@echo "Compiling APP $<..."
	@$(CXX) $(CXXFLAGS) $(INCLUDES_CORE) $(INCLUDES_APPS) -c $< -o $@

# Generic rule for actlib sources (must come after specific rules)
$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	@echo "Compiling $<..."
	@$(CXX) $(CXXFLAGS) $(INCLUDES_CORE) -c $< -o $@

# Executable targets
$(TEST_ACT_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/actlib/test/test_act.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT test executable created: $@"

$(TEST_ACT_CPU_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/actlib/test/test_act_cpu.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT_CPU test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT_CPU test executable created: $@"

$(TEST_ACT_CPU_F_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/actlib/test/test_act_cpu_f.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT_CPU (float) test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT_CPU (float) test executable created: $@"

$(TEST_ACT_ACCEL_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/actlib/test/test_act_accel.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT_Accelerate test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT_Accelerate test executable created: $@"

$(TEST_DICT_IO_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/actlib/test/test_dict_io.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking dictionary IO test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ Dictionary IO test executable created: $@"

$(PROFILE_ACT_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/actlib/test/profile_act.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT profiling executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT profiling executable created: $@"

$(PROFILE_ACT_MT_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/actlib/test/profile_act_mt.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT MT profiling executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT MT profiling executable created: $@"

# New: ACT_CPU multithreaded batch test
$(TEST_ACT_CPU_MT_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/actlib/test/test_act_cpu_mt.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT_CPU MT test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT_CPU MT test executable created: $@"

$(TEST_ACT_SYNTHETIC_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/actlib/test/test_act_synthetic.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking synthetic ACT test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ Synthetic ACT test executable created: $@"

$(TEST_ACT_CHALLENGING_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/actlib/test/test_act_challenging.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking challenging ACT test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ Challenging ACT test executable created: $@"

$(TEST_ACT_MLX_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/actlib/test/test_act_mlx.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT MLX test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT MLX test executable created: $@"

$(EEG_ACT_ANALYZER_TARGET): $(ACT_CORE_OBJECTS) $(EEG_ACT_ANALYZER_OBJECTS) $(ALGLIB_OBJECTS) $(LINENOISE_OBJECTS) | $(BINDIR)
	@echo "Linking EEG ACT analyzer executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ EEG ACT analyzer executable created: $@"

# Run targets
test-act-mlx: $(TEST_ACT_MLX_TARGET)
	@echo "Running ACT MLX test..."
	@./$(TEST_ACT_MLX_TARGET)

test-act-synthetic: $(TEST_ACT_SYNTHETIC_TARGET)
	@echo "Running synthetic ACT test..."
	@./$(TEST_ACT_SYNTHETIC_TARGET)

test-act-challenging: $(TEST_ACT_CHALLENGING_TARGET)
	@echo "Running challenging ACT test..."
	@./$(TEST_ACT_CHALLENGING_TARGET)

test-cpu-f: $(TEST_ACT_CPU_F_TARGET)
	@echo "Running ACT_CPU (float) test..."
	@./$(TEST_ACT_CPU_F_TARGET)

test-act-cpu-mt: $(TEST_ACT_CPU_MT_TARGET)
	@echo "Running ACT_CPU MT test..."
	@./$(TEST_ACT_CPU_MT_TARGET)

test-dict-io: $(TEST_DICT_IO_TARGET)
	@echo "💾 Running dictionary save/load test..."
	@./$(TEST_DICT_IO_TARGET)

test-stl: $(TEST_ACT_TARGET)
	@echo "🧪 Running stl ACT test..."
	@./$(TEST_ACT_TARGET)

test-cpu: $(TEST_ACT_CPU_TARGET)
	@echo "🧪 Running ACT_CPU test..."
	@./$(TEST_ACT_CPU_TARGET)

test-accel: $(TEST_ACT_ACCEL_TARGET)
	@echo "🧪 Running ACT_Accelerate test..."
	@./$(TEST_ACT_ACCEL_TARGET)

profile: $(PROFILE_ACT_TARGET)
	@echo "📊 Running ACT performance profiling..."
	@./$(PROFILE_ACT_TARGET)

profile-mt: $(PROFILE_ACT_MT_TARGET)
	@echo "📊 Running ACT MT performance profiling..."
	@./$(PROFILE_ACT_MT_TARGET)

eeg-act-analyzer: $(EEG_ACT_ANALYZER_TARGET)
	@echo "Running EEG ACT analyzer..."
	@./$(EEG_ACT_ANALYZER_TARGET)"

test: $(TEST_ACT_TARGET) $(TEST_ACT_CPU_TARGET) $(TEST_ACT_CPU_F_TARGET) $(TEST_ACT_ACCEL_TARGET) $(TEST_ACT_CPU_MT_TARGET) $(TEST_ACT_SYNTHETIC_TARGET) $(TEST_DICT_IO_TARGET) $(TEST_ACT_MLX_TARGET)
	@echo "🧪 Running all tests..."
	@./$(TEST_ACT_TARGET)
	@./$(TEST_ACT_CPU_TARGET)
	@./$(TEST_ACT_CPU_F_TARGET)
	@./$(TEST_ACT_ACCEL_TARGET)
	@./$(TEST_ACT_CPU_MT_TARGET)
	@./$(TEST_ACT_SYNTHETIC_TARGET)
	@./$(EEG_ACT_ANALYZER_TARGET)
	@./$(TEST_DICT_IO_TARGET)
	@./$(TEST_ACT_MLX_TARGET)

profile: $(PROFILE_ACT_TARGET) $(PROFILE_ACT_MT_TARGET)
	@echo "📊 Running all profiling..."
	@./$(PROFILE_ACT_TARGET)
	@./$(PROFILE_ACT_MT_TARGET)



# Utility targets
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf $(OBJDIR) $(BINDIR)
	@rm -f *.bin *.csv
	@echo "✅ Clean complete"

help:
	@echo "Adaptive Chirplet Transform (ACT) - Available Targets:"
	@echo ""
	@echo "Build Targets:"
	@echo "  all          - Build all executables"
	@echo "  test-act     - Build basic ACT test"
	@echo "  test-cpu     - Build ACT_CPU test"
	@echo "  test-accel   - Build ACT_Accelerate test"
	@echo "  test_act_mlx - Build ACT MLX test (falls back to CPU unless USE_MLX=1)"
	@echo "  profile      - Build performance profiling tool"
	@echo ""
	@echo "Run Targets:"
	@echo "  test         - Run basic ACT test"
	@echo "  profile      - Run performance profiling"
	@echo ""
	@echo "Utility Targets:"
	@echo "  clean        - Remove build artifacts"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - C++17 compatible compiler (g++ recommended)"
	@echo "  - ALGLIB (included in alglib/ directory)"
	@echo "  - macOS: Accelerate framework (automatic)"
	@echo "  - Linux: BLAS/LAPACK libraries"

# Phony targets
.PHONY: all test test-alglib-debug eeg-8s eeg-30s profile clean help eeg-analyzer eeg_act_analyzer test_act_mlx

# Dependency tracking
-include $(ACT_CORE_OBJECTS:.o=.d)
-include $(ALGLIB_OBJECTS:.o=.d)
