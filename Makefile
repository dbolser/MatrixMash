# Makefile for CUDA distance matrix computation

# CUDA settings
NVCC = nvcc
CUDA_ARCH = -arch=sm_86  # RTX 4060 Laptop (use sm_86, closest to 8.9 for CUDA 11.5)
CUDA_FLAGS = -O3 -Xcompiler -fPIC --compiler-options '-fPIC'
CXX_STD = -std=c++17

# CMake build directory for benchmarks
BUILD_DIR = build
CMAKE_BUILD_TYPE ?= Release

# Detect actual GPU architecture
detect_arch:
	@echo "Detecting GPU architecture..."
	@nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1

# Build shared library (original method, for Python wrapper)
libdistance_cuda.so: distance_cuda.cu
	@echo "Compiling CUDA kernel..."
	$(NVCC) $(CUDA_FLAGS) $(CUDA_ARCH) -shared -o $@ $<
	@echo "✓ Built $@"

libsnp_contribution.so: snp_contribution_kernel.cu
	@echo "Compiling SNP contribution kernel..."
	$(NVCC) $(CUDA_FLAGS) $(CUDA_ARCH) -shared -o $@ $<
	@echo "✓ Built $@"

snp_contribution_benchmark: snp_contribution_benchmark.cpp libsnp_contribution.so
	@echo "Building SNP contribution benchmark..."
	$(NVCC) $(CUDA_FLAGS) $(CUDA_ARCH) $(CXX_STD) -o $@ $< -L. -lsnp_contribution -Xlinker -rpath -Xlinker '$$ORIGIN'
	@echo "✓ Built $@"

# Alternative: auto-detect architecture (requires compute capability)
libdistance_cuda_auto.so: distance_cuda.cu
	@echo "Compiling with auto-detected architecture..."
	$(NVCC) $(CUDA_FLAGS) -arch=native -shared -o libdistance_cuda.so $<
	@echo "✓ Built libdistance_cuda.so"

# Test build
test: libdistance_cuda.so
	@echo "Running test..."
	uv run distance_cuda_wrapper.py genotype_data_genotypes.npy genotype_data_scores.npy

# Build benchmarks using CMake
benchmark: $(BUILD_DIR)/distance_cuda_benchmark
	@echo "Running benchmarks..."
	$(BUILD_DIR)/distance_cuda_benchmark

$(BUILD_DIR)/distance_cuda_benchmark: CMakeLists.txt distance_cuda.cu distance_cuda_benchmark.cpp
	@echo "Configuring CMake build..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DCUDA_ARCH=$(shell echo $(CUDA_ARCH) | sed 's/-arch=//') ..
	@echo "Building benchmarks..."
	@cd $(BUILD_DIR) && cmake --build . --target distance_cuda_benchmark
	@echo "✓ Built benchmark executable"

# Build everything (library + benchmarks)
all: libdistance_cuda.so $(BUILD_DIR)/distance_cuda_benchmark

# Clean
clean:
	rm -f libdistance_cuda.so *.o
	rm -rf $(BUILD_DIR)

# Clean everything including CMake cache
clean-all: clean
	rm -rf CMakeFiles/ CMakeCache.txt cmake_install.cmake _deps/
	rm -f *.a distance_cuda_benchmark

# Help
help:
	@echo "CUDA Distance Matrix Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make                  - Build libdistance_cuda.so (default sm_86 for RTX 4060)"
	@echo "  make libdistance_cuda_auto.so - Auto-detect GPU architecture"
	@echo "  make libsnp_contribution.so  - Build Phase 1 SNP contribution kernel"
	@echo "  make snp_contribution_benchmark - Build native benchmark harness"
	@echo "  make test            - Build and run test"
	@echo "  make benchmark       - Build and run Google Benchmark tests (uses CMake)"
	@echo "  make all             - Build library and benchmarks"
	@echo "  make detect_arch     - Show GPU compute capability"
	@echo "  make clean           - Remove build artifacts"
	@echo "  make clean-all       - Remove build artifacts and CMake cache"
	@echo ""
	@echo "Usage:"
	@echo "  make"
	@echo "  uv run distance_cuda_wrapper.py <genotypes.npy> <scores.npy>"
	@echo "  make benchmark       # Run C++ microbenchmarks"

.PHONY: test clean clean-all help detect_arch benchmark all
.DEFAULT_GOAL := libdistance_cuda.so
