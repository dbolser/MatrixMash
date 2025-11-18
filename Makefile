# Makefile for CUDA distance matrix computation

# CUDA settings
NVCC = nvcc
CUDA_ARCH = -arch=sm_86  # RTX 4060 Laptop (use sm_86, closest to 8.9 for CUDA 11.5)
CUDA_FLAGS = -O3 -Xcompiler -fPIC --compiler-options '-fPIC'

# Detect actual GPU architecture
detect_arch:
	@echo "Detecting GPU architecture..."
	@nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1

# Build shared library
libdistance_cuda.so: distance_cuda.cu
	@echo "Compiling CUDA kernel..."
	$(NVCC) $(CUDA_FLAGS) $(CUDA_ARCH) -shared -o $@ $<
	@echo "✓ Built $@"

libsnp_contribution.so: snp_contribution_kernel.cu
	@echo "Compiling SNP contribution kernel..."
	$(NVCC) $(CUDA_FLAGS) $(CUDA_ARCH) -shared -o $@ $<
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

# Clean
clean:
	rm -f libdistance_cuda.so libsnp_contribution.so *.o

# Help
help:
	@echo "CUDA Distance Matrix Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make                  - Build libdistance_cuda.so (default sm_86 for RTX 4060)"
	@echo "  make libdistance_cuda_auto.so - Auto-detect GPU architecture"
	@echo "  make libsnp_contribution.so  - Build Phase 1 SNP contribution kernel"
	@echo "  make test            - Build and run test"
	@echo "  make detect_arch     - Show GPU compute capability"
	@echo "  make clean           - Remove build artifacts"
	@echo ""
	@echo "Usage:"
	@echo "  make"
	@echo "  uv run distance_cuda_wrapper.py <genotypes.npy> <scores.npy>"

.PHONY: test clean help detect_arch
.DEFAULT_GOAL := libdistance_cuda.so
