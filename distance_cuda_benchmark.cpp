/*
 * Google Benchmark tests for CUDA distance matrix computation
 */

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <cstring>
#include <cstdio>

// Forward declaration of the CUDA function
extern "C" {
void compute_distance_cuda(
    const unsigned char* genotypes,
    const float* scores,
    float* distances,
    int M,
    int N
);
}

// Helper function to generate random test data
void generate_test_data(int M, int N, 
                       std::vector<unsigned char>& genotypes,
                       std::vector<float>& scores) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned char> geno_dist(0, 2);  // 0, 1, 2 (homozygous, heterozygous)
    std::uniform_real_distribution<float> score_dist(0.0f, 1.0f);
    
    genotypes.resize(M * N);
    scores.resize(M);
    
    for (int i = 0; i < M * N; i++) {
        genotypes[i] = geno_dist(gen);
    }
    
    for (int i = 0; i < M; i++) {
        scores[i] = score_dist(gen);
    }
}

// Benchmark function
static void BM_ComputeDistanceCUDA(benchmark::State& state) {
    int M = state.range(0);  // Number of genotypes
    int N = state.range(1);  // Number of individuals
    
    // Generate test data
    std::vector<unsigned char> genotypes;
    std::vector<float> scores;
    generate_test_data(M, N, genotypes, scores);
    
    // Allocate output
    std::vector<float> distances(N * N);
    
    // Warmup run (suppress printf output by redirecting stdout)
    FILE* null_out = fopen("/dev/null", "w");
    FILE* old_stdout = stdout;
    stdout = null_out;
    compute_distance_cuda(genotypes.data(), scores.data(), distances.data(), M, N);
    cudaDeviceSynchronize();
    stdout = old_stdout;
    fclose(null_out);
    
    // Benchmark loop (suppress printf output)
    null_out = fopen("/dev/null", "w");
    old_stdout = stdout;
    stdout = null_out;
    for (auto _ : state) {
        compute_distance_cuda(genotypes.data(), scores.data(), distances.data(), M, N);
        cudaDeviceSynchronize();
    }
    stdout = old_stdout;
    fclose(null_out);
    
    // Set counters
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(M) * N * N);
    state.SetBytesProcessed(
        state.iterations() * (
            static_cast<int64_t>(M) * N * sizeof(unsigned char) +  // genotypes
            static_cast<int64_t>(M) * sizeof(float) +               // scores
            static_cast<int64_t>(N) * N * sizeof(float)             // distances
        )
    );
}

// Register benchmarks with different problem sizes
BENCHMARK(BM_ComputeDistanceCUDA)
    ->Args({1000, 500})      // Small: 1k genotypes × 500 individuals
    ->Args({5000, 1000})      // Medium: 5k genotypes × 1k individuals
    ->Args({10000, 2000})     // Large: 10k genotypes × 2k individuals
    ->Args({10000, 5000})     // Very large: 10k genotypes × 5k individuals
    ->Args({100000, 2000})    // Many genotypes: 100k genotypes × 2k individuals
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Additional benchmark for scaling analysis
static void BM_ComputeDistanceCUDA_ScalingGenotypes(benchmark::State& state) {
    int M = state.range(0);
    int N = 1000;  // Fixed number of individuals
    
    std::vector<unsigned char> genotypes;
    std::vector<float> scores;
    generate_test_data(M, N, genotypes, scores);
    std::vector<float> distances(N * N);
    
    FILE* null_out = fopen("/dev/null", "w");
    FILE* old_stdout = stdout;
    stdout = null_out;
    compute_distance_cuda(genotypes.data(), scores.data(), distances.data(), M, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        compute_distance_cuda(genotypes.data(), scores.data(), distances.data(), M, N);
        cudaDeviceSynchronize();
    }
    stdout = old_stdout;
    fclose(null_out);
    
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(M) * N * N);
}

BENCHMARK(BM_ComputeDistanceCUDA_ScalingGenotypes)
    ->RangeMultiplier(2)
    ->Range(1000, 100000)  // 1k to 100k genotypes
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

static void BM_ComputeDistanceCUDA_ScalingIndividuals(benchmark::State& state) {
    int M = 10000;  // Fixed number of genotypes
    int N = state.range(0);
    
    std::vector<unsigned char> genotypes;
    std::vector<float> scores;
    generate_test_data(M, N, genotypes, scores);
    std::vector<float> distances(N * N);
    
    FILE* null_out = fopen("/dev/null", "w");
    FILE* old_stdout = stdout;
    stdout = null_out;
    compute_distance_cuda(genotypes.data(), scores.data(), distances.data(), M, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        compute_distance_cuda(genotypes.data(), scores.data(), distances.data(), M, N);
        cudaDeviceSynchronize();
    }
    stdout = old_stdout;
    fclose(null_out);
    
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(M) * N * N);
}

BENCHMARK(BM_ComputeDistanceCUDA_ScalingIndividuals)
    ->RangeMultiplier(2)
    ->Range(500, 10000)  // 500 to 10k individuals
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Complexity analysis benchmarks - these analyze scaling laws
// Theoretical complexity: O(M * N²)

// Analyze scaling with M (genotypes) - should be O(M) or linear
static void BM_Complexity_ScalingM(benchmark::State& state) {
    int M = state.range(0);
    int N = 2000;  // Fixed individuals
    
    std::vector<unsigned char> genotypes;
    std::vector<float> scores;
    generate_test_data(M, N, genotypes, scores);
    std::vector<float> distances(N * N);
    
    FILE* null_out = fopen("/dev/null", "w");
    FILE* old_stdout = stdout;
    stdout = null_out;
    compute_distance_cuda(genotypes.data(), scores.data(), distances.data(), M, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        compute_distance_cuda(genotypes.data(), scores.data(), distances.data(), M, N);
        cudaDeviceSynchronize();
    }
    stdout = old_stdout;
    fclose(null_out);
    
    // Set complexity to analyze - theoretical is O(M)
    state.SetComplexityN(M);
}

BENCHMARK(BM_Complexity_ScalingM)
    ->RangeMultiplier(2)
    ->Range(1000, 100000)  // 1k to 100k genotypes
    ->Complexity(benchmark::oN)  // Expected: O(N) linear in M
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Analyze scaling with N (individuals) - should be O(N²) quadratic
static void BM_Complexity_ScalingN(benchmark::State& state) {
    int M = 10000;  // Fixed genotypes
    int N = state.range(0);
    
    std::vector<unsigned char> genotypes;
    std::vector<float> scores;
    generate_test_data(M, N, genotypes, scores);
    std::vector<float> distances(N * N);
    
    FILE* null_out = fopen("/dev/null", "w");
    FILE* old_stdout = stdout;
    stdout = null_out;
    compute_distance_cuda(genotypes.data(), scores.data(), distances.data(), M, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        compute_distance_cuda(genotypes.data(), scores.data(), distances.data(), M, N);
        cudaDeviceSynchronize();
    }
    stdout = old_stdout;
    fclose(null_out);
    
    // Set complexity to analyze - theoretical is O(N²)
    state.SetComplexityN(N);
}

BENCHMARK(BM_Complexity_ScalingN)
    ->RangeMultiplier(2)
    ->Range(500, 10000)  // 500 to 10k individuals
    ->Complexity(benchmark::oNSquared)  // Expected: O(N²) quadratic in N
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Analyze overall scaling with problem size (M*N²) - should be O(M*N²)
static void BM_Complexity_OverallScaling(benchmark::State& state) {
    // Use state.range(0) as a scale factor
    // Scale both M and N proportionally to maintain aspect ratio
    int base_M = 1000;
    int base_N = 500;
    int scale = state.range(0);
    
    int M = base_M * scale;
    int N = base_N * scale;
    
    std::vector<unsigned char> genotypes;
    std::vector<float> scores;
    generate_test_data(M, N, genotypes, scores);
    std::vector<float> distances(N * N);
    
    FILE* null_out = fopen("/dev/null", "w");
    FILE* old_stdout = stdout;
    stdout = null_out;
    compute_distance_cuda(genotypes.data(), scores.data(), distances.data(), M, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        compute_distance_cuda(genotypes.data(), scores.data(), distances.data(), M, N);
        cudaDeviceSynchronize();
    }
    stdout = old_stdout;
    fclose(null_out);
    
    // Complexity is measured against M*N² (theoretical work)
    int64_t problem_size = static_cast<int64_t>(M) * N * N;
    state.SetComplexityN(problem_size);
}

BENCHMARK(BM_Complexity_OverallScaling)
    ->RangeMultiplier(2)
    ->Range(1, 20)  // Scale factor: 1x to 20x
    ->Complexity(benchmark::oN)  // Expected: O(N) linear in problem size M*N²
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Additional complexity tests - test multiple complexity hypotheses
// These help verify the scaling by testing against different complexity classes

static void BM_Complexity_ScalingM_LogN(benchmark::State& state) {
    int M = state.range(0);
    int N = 2000;
    
    std::vector<unsigned char> genotypes;
    std::vector<float> scores;
    generate_test_data(M, N, genotypes, scores);
    std::vector<float> distances(N * N);
    
    FILE* null_out = fopen("/dev/null", "w");
    FILE* old_stdout = stdout;
    stdout = null_out;
    compute_distance_cuda(genotypes.data(), scores.data(), distances.data(), M, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        compute_distance_cuda(genotypes.data(), scores.data(), distances.data(), M, N);
        cudaDeviceSynchronize();
    }
    stdout = old_stdout;
    fclose(null_out);
    
    state.SetComplexityN(M);
}

// Test if it's actually O(N log N) instead of O(N) - should have worse fit
BENCHMARK(BM_Complexity_ScalingM_LogN)
    ->RangeMultiplier(2)
    ->Range(1000, 100000)
    ->Complexity(benchmark::oNLogN)  // Test O(N log N) hypothesis
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

static void BM_Complexity_ScalingN_Cubic(benchmark::State& state) {
    int M = 10000;
    int N = state.range(0);
    
    std::vector<unsigned char> genotypes;
    std::vector<float> scores;
    generate_test_data(M, N, genotypes, scores);
    std::vector<float> distances(N * N);
    
    FILE* null_out = fopen("/dev/null", "w");
    FILE* old_stdout = stdout;
    stdout = null_out;
    compute_distance_cuda(genotypes.data(), scores.data(), distances.data(), M, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        compute_distance_cuda(genotypes.data(), scores.data(), distances.data(), M, N);
        cudaDeviceSynchronize();
    }
    stdout = old_stdout;
    fclose(null_out);
    
    state.SetComplexityN(N);
}

// Test if it's actually O(N³) instead of O(N²) - should have worse fit
BENCHMARK(BM_Complexity_ScalingN_Cubic)
    ->RangeMultiplier(2)
    ->Range(500, 10000)
    ->Complexity(benchmark::oNCubed)  // Test O(N³) hypothesis
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();

