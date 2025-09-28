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
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Wuninitialized -g -MMD -MP -I. -Ialglib/alglib-cpp/src -Ilinenoise -Ithird_party/spdlog/include -Ithird_party/oscpp/include
CFLAGS = -O3 -Wall -Wextra -g -MMD -MP -I. -Ialglib/alglib-cpp/src -Ilinenoise -Ithird_party/spdlog/include -Ithird_party/oscpp/include
LDFLAGS = -lm -pthread

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

# Core ACT Sources (legacy SIMD/MT and benchmarks removed)
ACT_CORE_SOURCES = ACT.cpp ACT_CPU.cpp ACT_Accelerate.cpp ACT_MLX.cpp

# Test Sources
TEST_ACT_SOURCES = test_act.cpp
TEST_ACT_CPU_SOURCES = test_act_cpu.cpp
TEST_ACT_CPU_F_SOURCES = test_act_cpu_f.cpp
TEST_ACT_ACCEL_SOURCES = test_act_accel.cpp
TEST_ACT_CPU_MT_SOURCES = test_act_cpu_mt.cpp
PROFILE_ACT_SOURCES = profile_act.cpp
PROFILE_ACT_MT_SOURCES = profile_act_mt.cpp

# Linenoise library
LINENOISE_SOURCES = linenoise/linenoise.c
LINENOISE_OBJECTS = $(OBJDIR)/linenoise/linenoise.o

# ALGLIB Dependencies (Essential Components)
ALGLIB_SOURCES = \
	alglib/alglib-cpp/src/ap.cpp \
	alglib/alglib-cpp/src/alglibinternal.cpp \
	alglib/alglib-cpp/src/alglibmisc.cpp \
	alglib/alglib-cpp/src/optimization.cpp \
	alglib/alglib-cpp/src/linalg.cpp \
	alglib/alglib-cpp/src/solvers.cpp \
	alglib/alglib-cpp/src/dataanalysis.cpp \
	alglib/alglib-cpp/src/interpolation.cpp \
	alglib/alglib-cpp/src/specialfunctions.cpp \
	alglib/alglib-cpp/src/statistics.cpp \
	alglib/alglib-cpp/src/fasttransforms.cpp \
	alglib/alglib-cpp/src/integration.cpp \
	alglib/alglib-cpp/src/diffequations.cpp

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
    LDFLAGS += -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework QuartzCore
    ifneq ($(MLX_LIB),)
        LDFLAGS += -L$(MLX_LIB)
    endif
    ifneq ($(MLX_LINK),)
        LDFLAGS += $(MLX_LINK)
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
TEST_ACT_MLX_TARGET = $(BINDIR)/test_act_mlx
PROFILE_ACT_MT_TARGET = $(BINDIR)/profile_act_mt
EEG_ACT_ANALYZER_TARGET = $(BINDIR)/eeg_act_analyzer
TEST_DICT_IO_TARGET = $(BINDIR)/test_dict_io

# Default target
all: $(TEST_ACT_TARGET) $(TEST_ACT_CPU_TARGET) $(TEST_ACT_CPU_F_TARGET) $(TEST_ACT_ACCEL_TARGET) $(TEST_ACT_CPU_MT_TARGET) $(PROFILE_ACT_TARGET) $(PROFILE_ACT_MT_TARGET) $(TEST_ACT_SYNTHETIC_TARGET) $(EEG_ACT_ANALYZER_TARGET) $(TEST_DICT_IO_TARGET) $(TEST_ACT_MLX_TARGET)

# Create directories
$(OBJDIR):
	@mkdir -p $(OBJDIR)
	@mkdir -p $(OBJDIR)/alglib/alglib-cpp/src
	@mkdir -p $(OBJDIR)/linenoise

$(BINDIR):
	@mkdir -p $(BINDIR)

# Object file compilation rules
$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	@echo "Compiling $<..."
	@$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/linenoise/linenoise.o: linenoise/linenoise.c | $(OBJDIR)
	@echo "Compiling LINENOISE $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/alglib/alglib-cpp/src/%.o: alglib/alglib-cpp/src/%.cpp | $(OBJDIR)
	@echo "Compiling ALGLIB $<..."
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Executable targets
$(TEST_ACT_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/test_act.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT test executable created: $@"

$(TEST_ACT_CPU_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/test_act_cpu.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT_CPU test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT_CPU test executable created: $@"

$(TEST_ACT_CPU_F_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/test_act_cpu_f.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT_CPU (float) test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT_CPU (float) test executable created: $@"

$(TEST_ACT_ACCEL_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/test_act_accel.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT_Accelerate test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT_Accelerate test executable created: $@"

$(TEST_DICT_IO_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/test_dict_io.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking dictionary IO test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ Dictionary IO test executable created: $@"

$(PROFILE_ACT_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/profile_act.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT profiling executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT profiling executable created: $@"

$(PROFILE_ACT_MT_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/profile_act_mt.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT MT profiling executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT MT profiling executable created: $@"

# New: ACT_CPU multithreaded batch test
$(TEST_ACT_CPU_MT_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/test_act_cpu_mt.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT_CPU MT test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT_CPU MT test executable created: $@"

$(TEST_ACT_SYNTHETIC_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/test_act_synthetic.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking synthetic ACT test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ Synthetic ACT test executable created: $@"

 

$(TEST_ACT_MLX_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/test_act_mlx.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT MLX test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT MLX test executable created: $@"

# Convenience alias to build MLX test (ensures correct target path is used)
test_act_mlx: $(TEST_ACT_MLX_TARGET)
	@echo "Built $(TEST_ACT_MLX_TARGET)"

# Convenience alias for synthetic test
test_act_synthetic: $(TEST_ACT_SYNTHETIC_TARGET)
	@echo "Built $(TEST_ACT_SYNTHETIC_TARGET)"

# Convenience aliases for ACT_CPU_f test
test-cpu-f: $(TEST_ACT_CPU_F_TARGET)
	@echo "Built $(TEST_ACT_CPU_F_TARGET)"

test_act_cpu_f: $(TEST_ACT_CPU_F_TARGET)
	@echo "Built $(TEST_ACT_CPU_F_TARGET)"

# Convenience alias for ACT_CPU MT test
test_act_cpu_mt: $(TEST_ACT_CPU_MT_TARGET)
	@echo "Built $(TEST_ACT_CPU_MT_TARGET)"

$(EEG_ACT_ANALYZER_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/eeg_act_analyzer.o $(OBJDIR)/muse_osc_receiver.o $(ALGLIB_OBJECTS) $(LINENOISE_OBJECTS) | $(BINDIR)
	@echo "Linking EEG ACT analyzer executable..."
	@$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✅ EEG ACT analyzer executable created: $@"

eeg-analyzer: $(EEG_ACT_ANALYZER_TARGET)

eeg_act_analyzer: eeg-analyzer

# Run targets

test-dict-io: $(TEST_DICT_IO_TARGET)
	@echo "💾 Running dictionary save/load test..."
	@./$(TEST_DICT_IO_TARGET)

test: $(TEST_ACT_TARGET)
	@echo "🧪 Running basic ACT test..."
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
