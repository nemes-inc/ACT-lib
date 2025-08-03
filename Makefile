# Adaptive Chirplet Transform (ACT) - C++ Implementation
# Professional Makefile for Public Release
# 
# Usage:
#   make all          - Build all targets
#   make test         - Run basic ACT test
#   make eeg-8s       - Run 8-second EEG gamma analysis
#   make eeg-30s      - Run 30-second EEG gamma analysis
#   make profile      - Run performance profiling
#   make clean        - Clean build artifacts
#   make help         - Show available targets

# Compiler Configuration
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -g -I. -Ialglib/alglib-cpp/src
LDFLAGS = -lm -pthread

# Platform-specific optimizations
ifeq ($(shell uname),Darwin)
    CXXFLAGS += -march=native
    LDFLAGS += -framework Accelerate
else ifeq ($(shell uname),Linux)
    CXXFLAGS += -march=native
    LDFLAGS += -lblas -llapack
endif

# Directories
OBJDIR = obj
BINDIR = bin

# Core ACT Sources
ACT_CORE_SOURCES = ACT.cpp ACT_SIMD.cpp ACT_SIMD_MultiThreaded.cpp ACT_multithreaded.cpp ACT_Benchmark.cpp

# Test Sources
TEST_ACT_SOURCES = test_act.cpp
TEST_EEG_GAMMA_SOURCES = test_eeg_gamma.cpp
TEST_EEG_GAMMA_8S_SOURCES = test_eeg_gamma_8s.cpp
TEST_EEG_GAMMA_30S_SOURCES = test_eeg_gamma_30s.cpp
TEST_SIMD_SOURCES = test_simd.cpp
TEST_SIMD_MT_SOURCES = test_simd_multithreaded.cpp
PROFILE_ACT_SOURCES = profile_act.cpp

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

# Executables
TEST_ACT_TARGET = $(BINDIR)/test_act
TEST_EEG_GAMMA_TARGET = $(BINDIR)/test_eeg_gamma
TEST_EEG_GAMMA_8S_TARGET = $(BINDIR)/test_eeg_gamma_8s
TEST_EEG_GAMMA_30S_TARGET = $(BINDIR)/test_eeg_gamma_30s
TEST_SIMD_TARGET = $(BINDIR)/test_simd
TEST_SIMD_MT_TARGET = $(BINDIR)/test_simd_multithreaded
PROFILE_ACT_TARGET = $(BINDIR)/profile_act

# Default target
all: $(TEST_ACT_TARGET) $(TEST_EEG_GAMMA_8S_TARGET) $(TEST_EEG_GAMMA_30S_TARGET) $(TEST_SIMD_TARGET) $(PROFILE_ACT_TARGET)

# Create directories
$(OBJDIR):
	@mkdir -p $(OBJDIR)
	@mkdir -p $(OBJDIR)/alglib/alglib-cpp/src

$(BINDIR):
	@mkdir -p $(BINDIR)

# Object file compilation rules
$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	@echo "Compiling $<..."
	@$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/alglib/alglib-cpp/src/%.o: alglib/alglib-cpp/src/%.cpp | $(OBJDIR)
	@echo "Compiling ALGLIB $<..."
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Executable targets
$(TEST_ACT_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/test_act.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT test executable created: $@"

$(TEST_EEG_GAMMA_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/test_eeg_gamma.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking EEG gamma analysis executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ EEG gamma analysis executable created: $@"

$(TEST_EEG_GAMMA_8S_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/test_eeg_gamma_8s.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking 8s EEG gamma analysis executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ 8s EEG gamma analysis executable created: $@"

$(TEST_EEG_GAMMA_30S_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/test_eeg_gamma_30s.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking 30s EEG gamma analysis executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ 30s EEG gamma analysis executable created: $@"

$(TEST_SIMD_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/test_simd.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking SIMD test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ SIMD test executable created: $@"

$(TEST_SIMD_MT_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/test_simd_multithreaded.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking SIMD multithreaded test executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ SIMD multithreaded test executable created: $@"

$(PROFILE_ACT_TARGET): $(ACT_CORE_OBJECTS) $(OBJDIR)/profile_act.o $(ALGLIB_OBJECTS) | $(BINDIR)
	@echo "Linking ACT profiling executable..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✅ ACT profiling executable created: $@"

# Run targets
test: $(TEST_ACT_TARGET)
	@echo "🧪 Running basic ACT test..."
	@./$(TEST_ACT_TARGET)

eeg-8s: $(TEST_EEG_GAMMA_8S_TARGET)
	@echo "🧠 Running 8-second EEG gamma analysis..."
	@./$(TEST_EEG_GAMMA_8S_TARGET)

eeg-30s: $(TEST_EEG_GAMMA_30S_TARGET)
	@echo "🧠 Running 30-second EEG gamma analysis..."
	@./$(TEST_EEG_GAMMA_30S_TARGET)

simd: $(TEST_SIMD_TARGET)
	@echo "⚡ Running SIMD performance test..."
	@./$(TEST_SIMD_TARGET)

simd-mt: $(TEST_SIMD_MT_TARGET)
	@echo "⚡ Running SIMD multithreaded test..."
	@./$(TEST_SIMD_MT_TARGET)

profile: $(PROFILE_ACT_TARGET)
	@echo "📊 Running ACT performance profiling..."
	@./$(PROFILE_ACT_TARGET)

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
	@echo "  eeg-8s       - Build 8-second EEG analysis"
	@echo "  eeg-30s      - Build 30-second EEG analysis"
	@echo "  simd         - Build SIMD performance test"
	@echo "  profile      - Build performance profiling tool"
	@echo ""
	@echo "Run Targets:"
	@echo "  test         - Run basic ACT test"
	@echo "  eeg-8s       - Run 8-second EEG gamma analysis"
	@echo "  eeg-30s      - Run 30-second EEG gamma analysis"
	@echo "  simd         - Run SIMD performance test"
	@echo "  simd-mt      - Run SIMD multithreaded test"
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
.PHONY: all test eeg-8s eeg-30s simd simd-mt profile clean help

# Dependency tracking
-include $(ACT_CORE_OBJECTS:.o=.d)
-include $(ALGLIB_OBJECTS:.o=.d)
