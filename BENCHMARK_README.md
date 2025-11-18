# C++ Benchmarking with Google Benchmark

This project uses [Google Benchmark](https://github.com/google/benchmark) for C++ microbenchmarking of the CUDA distance matrix computation kernels.

## Prerequisites

- CMake 3.18 or higher
- CUDA toolkit
- C++17 compatible compiler
- NVIDIA GPU with CUDA support

## Building Benchmarks

### Using CMake directly (recommended):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=86
cmake --build build --target distance_cuda_benchmark
```

**Notes**

- Always configure out of source (`build/`) to keep the repo clean.
- Pass the numeric compute capability (`86`, `75`, `89`, …) to `CUDA_ARCH`.

### Using Makefile:

```bash
make benchmark              # uses CUDA_ARCH=-arch=sm_86 by default
```

This will:
1. Configure CMake (automatically downloads Google Benchmark)
2. Build the benchmark executable
3. Run the benchmarks

## Building the CUDA Library

Use the Makefile for the shared library that the Python wrapper consumes:

```bash
make                              # builds libdistance_cuda.so with sm_86
make libdistance_cuda_auto.so     # lets NVCC auto-detect the architecture
```

The same target can be produced via CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=86
cmake --build build --target distance_cuda_shared
```

## Running Benchmarks

After building, run the benchmark executable:

```bash
./build/distance_cuda_benchmark
```

### Benchmark Options

Google Benchmark supports many command-line options:

```bash
# Run specific benchmarks
./build/distance_cuda_benchmark --benchmark_filter="BM_ComputeDistanceCUDA"

# Run only complexity analysis benchmarks
./build/distance_cuda_benchmark --benchmark_filter="BM_Complexity"

# Run with more iterations (better for complexity analysis)
./build/distance_cuda_benchmark --benchmark_min_time=2.0

# Output JSON results
./build/distance_cuda_benchmark --benchmark_format=json > results.json

# Compare with previous run
./build/distance_cuda_benchmark --benchmark_format=json | \
  benchmark/tools/compare.py benchmarks < baseline.json

# Show help
./build/distance_cuda_benchmark --help
```

### Complexity Analysis

The complexity benchmarks analyze scaling laws to verify theoretical complexity:

- **Theoretical**: O(M × N²) - linear in genotypes, quadratic in individuals
- **Measured**: Google Benchmark fits the actual scaling and reports:
  - How well it matches expected complexity (RMS error)
  - Actual Big-O notation if different from theoretical
  - Performance characteristics on GPU vs. theoretical

This helps identify:
- Whether GPU implementation matches theoretical complexity
- Memory bandwidth bottlenecks (may show sub-linear scaling)
- Kernel launch overhead (may affect small problem sizes)
- Cache effects (may show better than O(N²) for certain ranges)

## Benchmark Tests

The benchmark suite includes:

1. **BM_ComputeDistanceCUDA**: Tests various problem sizes:
   - Small: 1,000 genotypes × 500 individuals
   - Medium: 5,000 genotypes × 1,000 individuals
   - Large: 10,000 genotypes × 2,000 individuals
   - Very large: 10,000 genotypes × 5,000 individuals
   - Many genotypes: 100,000 genotypes × 2,000 individuals

2. **BM_ComputeDistanceCUDA_ScalingGenotypes**: Scaling analysis with fixed individuals (1,000) and varying genotypes (1k to 100k)

3. **BM_ComputeDistanceCUDA_ScalingIndividuals**: Scaling analysis with fixed genotypes (10k) and varying individuals (500 to 10k)

4. **Complexity Analysis Benchmarks**: Analyze scaling laws to verify theoretical complexity:
   - **BM_Complexity_ScalingM**: Tests if scaling with M (genotypes) is O(M) linear
   - **BM_Complexity_ScalingN**: Tests if scaling with N (individuals) is O(N²) quadratic
   - **BM_Complexity_OverallScaling**: Tests overall scaling with problem size M×N²
   - **BM_Complexity_ScalingM_LogN**: Tests alternative hypothesis O(M log M) - should have worse fit
   - **BM_Complexity_ScalingN_Cubic**: Tests alternative hypothesis O(N³) - should have worse fit

## Output Metrics

Each benchmark reports:
- **Time**: Wall-clock time per iteration
- **Items Processed**: Number of genotype comparisons (M × N²)
- **Bytes Processed**: Total memory transferred (genotypes + scores + distances)
- **Throughput**: Computations per second

For complexity analysis benchmarks, additional metrics include:
- **Complexity**: Fitted complexity (e.g., O(N), O(N²), O(N log N))
- **RMS**: Root mean square error of the fit
- **Big-O**: Automatically determined Big-O notation

## Example Output

```
Run on (8 X 2400 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 256 KiB (x8)
  L3 Unified 16384 KiB (x1)
Load Average: 0.52, 0.58, 0.59
--------------------------------------------------------------------
Benchmark                          Time             CPU   Iterations
--------------------------------------------------------------------
BM_ComputeDistanceCUDA/1000/500   1.234 ms        1.234 ms        567
BM_ComputeDistanceCUDA/5000/1000  12.34 ms       12.34 ms         57
BM_ComputeDistanceCUDA/10000/2000 45.67 ms       45.67 ms         15
```

### Complexity Analysis Output

The complexity benchmarks show how performance scales:

```
BM_Complexity_ScalingM/1000       0.123 ms        0.123 ms     5678
BM_Complexity_ScalingM/2000       0.246 ms        0.246 ms     2845
BM_Complexity_ScalingM/4000       0.492 ms        0.492 ms     1423
BM_Complexity_ScalingM/8000       0.984 ms        0.984 ms      712
BM_Complexity_ScalingM/16000      1.968 ms        1.968 ms      356
BM_Complexity_ScalingM_BigO       0.123 N         0.123 N

BM_Complexity_ScalingN/500        0.123 ms        0.123 ms     5678
BM_Complexity_ScalingN/1000       0.492 ms        0.492 ms     1423
BM_Complexity_ScalingN/2000       1.968 ms        1.968 ms      356
BM_Complexity_ScalingN/4000       7.872 ms        7.872 ms       89
BM_Complexity_ScalingN_BigO       0.000 N^2       0.000 N^2
```

This shows:
- **Scaling with M**: Linear O(N) - doubling M doubles time
- **Scaling with N**: Quadratic O(N²) - doubling N quadruples time
- **Overall**: Confirms theoretical O(M × N²) complexity

## Integration with CI/CD

To run benchmarks in CI:

```bash
# Build and run, output JSON
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target distance_cuda_benchmark
./distance_cuda_benchmark --benchmark_format=json > benchmark_results.json
```

## Customizing CUDA Architecture

Set the CUDA architecture when configuring CMake:

```bash
cmake -S . -B build -DCUDA_ARCH=86   # For RTX 4060 (sm_86)
cmake -S . -B build -DCUDA_ARCH=75   # For RTX 2000 series (sm_75)
cmake -S . -B build -DCUDA_ARCH=89   # For RTX 5000 series (sm_89)
```

Or use the Makefile:

```bash
make benchmark CUDA_ARCH=-arch=sm_89
```

## Troubleshooting

### Google Benchmark not downloading
- Ensure you have internet access (CMake downloads it via FetchContent)
- Check CMake version (3.18+ required)

### CUDA errors
- Verify CUDA toolkit is installed: `nvcc --version`
- Check GPU is available: `nvidia-smi`
- Ensure CUDA_ARCH matches your GPU

### Build errors
- Clean and rebuild: `make clean-all && make benchmark`
- Check CMake output for specific errors

## References

- [Google Benchmark Documentation](https://github.com/google/benchmark)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

