#!/usr/bin/env python3
"""Test GPU scaling across different M and N."""

import numpy as np
import time
import sys

# Import from distance_torch
sys.path.insert(0, '.')
from distance_torch import compute_distance_torch

def generate_data(M, N, seed=42):
    """Generate test data."""
    np.random.seed(seed)
    genotypes = np.random.choice([0, 1, 2, 9], size=(M, N), p=[0.70, 0.20, 0.05, 0.05]).astype(np.uint8)
    scores = np.random.gamma(shape=2.0, scale=0.2, size=M).astype(np.float32)
    scores = np.clip(scores, 0.0, 2.0)
    return genotypes, scores

def benchmark(M, N, chunk_size, device='cuda'):
    """Run benchmark and return timing."""
    import torch

    print(f"\nTesting M={M:,}, N={N:,}, chunk_size={chunk_size}")
    print("-" * 60)

    # Clear GPU cache before test
    torch.cuda.empty_cache()

    # Generate data
    genotypes, scores = generate_data(M, N)

    # Benchmark
    start = time.time()
    distances = compute_distance_torch(genotypes, scores, chunk_size, device)
    elapsed = time.time() - start

    # Calculate metrics
    pairs = N * (N + 1) // 2
    throughput = pairs / elapsed
    comparisons_per_sec = pairs * M / elapsed

    print(f"Time: {elapsed:.3f} seconds")
    print(f"Throughput: {throughput:.0f} pairs/sec")
    print(f"Comparisons: {comparisons_per_sec/1e6:.1f} M/sec")

    # Clean up
    del genotypes, scores, distances
    torch.cuda.empty_cache()

    return {
        'M': M,
        'N': N,
        'chunk_size': chunk_size,
        'time': elapsed,
        'pairs': pairs,
        'throughput': throughput,
        'comparisons_per_sec': comparisons_per_sec
    }

if __name__ == "__main__":
    import torch

    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    results = []

    # Test 1: Vary N, keep M constant (test pair scaling)
    print("\n" + "="*60)
    print("TEST 1: Varying N (individuals), M=10,000 fixed")
    print("="*60)

    M = 10_000
    chunk_size = 500

    for N in [1000, 2000, 3000, 4000, 5000]:
        result = benchmark(M, N, chunk_size)
        results.append(result)

    # Test 2: Vary M, keep N constant (test genotype scaling)
    print("\n" + "="*60)
    print("TEST 2: Varying M (genotypes), N=4,000 fixed")
    print("="*60)

    N = 4_000

    for M in [10_000, 25_000, 50_000, 75_000, 100_000]:
        # Use smaller chunks for larger M to avoid OOM
        if M <= 10_000:
            chunk_size = 500
        elif M <= 25_000:
            chunk_size = 300
        elif M <= 50_000:
            chunk_size = 200
        else:
            chunk_size = 150
        result = benchmark(M, N, chunk_size)
        results.append(result)

    # Test 3: Different chunk sizes (N=3000, M=10000)
    print("\n" + "="*60)
    print("TEST 3: Varying chunk_size, M=10,000, N=3,000")
    print("="*60)

    M = 10_000
    N = 3_000

    for chunk_size in [100, 200, 300, 500, 750, 1000]:
        result = benchmark(M, N, chunk_size)
        results.append(result)

    # Analysis
    print("\n" + "="*60)
    print("SCALING ANALYSIS")
    print("="*60)

    # Test 1 results: N scaling
    print("\nN Scaling (M=10,000 fixed):")
    print(f"{'N':<10} {'Time (s)':<12} {'Pairs':<15} {'Time/Pair (µs)':<15}")
    print("-" * 60)

    test1_results = [r for r in results if r['M'] == 10_000 and r['chunk_size'] == 500]
    for r in test1_results:
        time_per_pair = r['time'] / r['pairs'] * 1e6
        print(f"{r['N']:<10} {r['time']:<12.3f} {r['pairs']:<15,} {time_per_pair:<15.3f}")

    # Check if O(N^2)
    if len(test1_results) >= 2:
        r1, r2 = test1_results[0], test1_results[-1]
        expected_ratio = (r2['N'] / r1['N']) ** 2
        actual_ratio = r2['time'] / r1['time']
        print(f"\nExpected O(N²) ratio: {expected_ratio:.2f}x")
        print(f"Actual ratio: {actual_ratio:.2f}x")
        print(f"Efficiency: {actual_ratio/expected_ratio*100:.1f}%")

    # Test 2 results: M scaling
    print("\n" + "-"*60)
    print("M Scaling (N=2,000 fixed):")
    print(f"{'M':<10} {'Time (s)':<12} {'Comparisons/sec':<20}")
    print("-" * 60)

    test2_results = [r for r in results if r['N'] == 2_000]
    for r in test2_results:
        print(f"{r['M']:<10,} {r['time']:<12.3f} {r['comparisons_per_sec']/1e6:<20.1f} M/sec")

    # Check if O(M) with constant throughput
    if len(test2_results) >= 2:
        r1, r2 = test2_results[0], test2_results[-1]
        expected_ratio = r2['M'] / r1['M']
        actual_ratio = r2['time'] / r1['time']
        print(f"\nExpected O(M) ratio: {expected_ratio:.2f}x")
        print(f"Actual ratio: {actual_ratio:.2f}x")
        print(f"Efficiency: {actual_ratio/expected_ratio*100:.1f}%")

        throughput1 = r1['comparisons_per_sec']
        throughput2 = r2['comparisons_per_sec']
        print(f"Throughput change: {throughput2/throughput1*100:.1f}%")

    # Test 3 results: Chunk size
    print("\n" + "-"*60)
    print("Chunk Size Impact (M=10,000, N=3,000):")
    print(f"{'Chunk':<10} {'Time (s)':<12} {'Throughput':<20}")
    print("-" * 60)

    test3_results = [r for r in results if r['M'] == 10_000 and r['N'] == 3_000]
    best_chunk = min(test3_results, key=lambda r: r['time'])

    for r in test3_results:
        marker = " ←BEST" if r['chunk_size'] == best_chunk['chunk_size'] else ""
        print(f"{r['chunk_size']:<10} {r['time']:<12.3f} {r['comparisons_per_sec']/1e6:<20.1f} M/sec{marker}")

    # Final extrapolation
    print("\n" + "="*60)
    print("EXTRAPOLATION TO FULL DATASET (1M × 50k)")
    print("="*60)

    # Use the largest test for most accurate prediction
    largest_test = max(results[:10], key=lambda r: r['M'] * r['pairs'])

    M_full = 1_000_000
    N_full = 50_000
    pairs_full = N_full * (N_full + 1) // 2

    # Method 1: Scale from largest test
    scale_factor = (N_full / largest_test['N']) ** 2 * (M_full / largest_test['M'])
    estimated_time = largest_test['time'] * scale_factor

    print(f"\nMethod 1: Direct scaling from M={largest_test['M']:,}, N={largest_test['N']:,}")
    print(f"  Base time: {largest_test['time']:.3f} seconds")
    print(f"  Scale factor: {scale_factor:.1f}x")
    print(f"  Estimated: {estimated_time:.1f} seconds = {estimated_time/3600:.2f} hours = {estimated_time/86400:.2f} days")

    # Method 2: Use average throughput
    avg_throughput = np.mean([r['comparisons_per_sec'] for r in results[:10]])
    total_comparisons = pairs_full * M_full
    estimated_time2 = total_comparisons / avg_throughput

    print(f"\nMethod 2: Average throughput")
    print(f"  Avg throughput: {avg_throughput/1e6:.1f} M comparisons/sec")
    print(f"  Total comparisons: {total_comparisons:.2e}")
    print(f"  Estimated: {estimated_time2:.1f} seconds = {estimated_time2/3600:.2f} hours = {estimated_time2/86400:.2f} days")

    # Conservative estimate
    conservative = max(estimated_time, estimated_time2)
    print(f"\nConservative estimate: {conservative/3600:.2f} hours = {conservative/86400:.2f} days")

    # GPU vs CPU comparison
    cpu_time_days = 9  # From CPU benchmarks
    gpu_speedup = (cpu_time_days * 86400) / conservative
    print(f"\nGPU speedup vs CPU: {gpu_speedup:.1f}x")
