#!/usr/bin/env python3
"""Test CPU core scaling."""

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads
import time
import os

@njit(parallel=True, fastmath=True)
def compute_distance_numba_cpu(genotypes, scores, distances):
    """Numba JIT-compiled CPU implementation with parallel loops."""
    M, N = genotypes.shape
    s2 = scores * scores

    for i in prange(N):
        for j in range(i, N):
            dist_sum = 0.0
            for m in range(M):
                geno_i = genotypes[m, i]
                geno_j = genotypes[m, j]

                if geno_i == 9 or geno_j == 9:
                    continue

                diff = abs(int(geno_i) - int(geno_j))

                if diff == 1:
                    dist_sum += s2[m]
                elif diff == 2:
                    dist_sum += 4.0 * s2[m]

            distances[i, j] = dist_sum
            distances[j, i] = dist_sum


def benchmark_threads(genotypes, scores, num_threads):
    """Benchmark with specific thread count."""
    M, N = genotypes.shape
    distances = np.zeros((N, N), dtype=np.float32)

    # Set thread count
    set_num_threads(num_threads)

    # Warmup if first run
    if num_threads == 1:
        small_geno = genotypes[:100, :10]
        small_scores = scores[:100]
        small_dist = np.zeros((10, 10), dtype=np.float32)
        compute_distance_numba_cpu(small_geno, small_scores, small_dist)

    # Benchmark
    start = time.time()
    compute_distance_numba_cpu(genotypes, scores, distances)
    elapsed = time.time() - start

    return elapsed, distances


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test CPU core scaling")
    parser.add_argument("genotypes_file", help="Path to genotypes .npy file")
    parser.add_argument("scores_file", help="Path to scores .npy file")

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    genotypes = np.load(args.genotypes_file)
    scores = np.load(args.scores_file)

    M, N = genotypes.shape
    print(f"Dataset: {M:,} genotypes × {N:,} individuals\n")

    # Get max threads
    max_threads = os.cpu_count()
    print(f"System has {max_threads} CPU cores\n")

    # Test different thread counts
    thread_counts = [1, 2, 4, 8, 16, 32]
    if max_threads and max_threads not in thread_counts:
        thread_counts.append(max_threads)
    thread_counts = sorted([t for t in thread_counts if t <= (max_threads or 64)])

    results = []

    print("=" * 70)
    print(f"{'Threads':<10} {'Time (s)':<12} {'Speedup':<12} {'Efficiency':<12}")
    print("=" * 70)

    baseline_time = None

    for num_threads in thread_counts:
        elapsed, distances = benchmark_threads(genotypes, scores, num_threads)

        if baseline_time is None:
            baseline_time = elapsed
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed

        efficiency = speedup / num_threads * 100

        print(f"{num_threads:<10} {elapsed:<12.3f} {speedup:<12.2f}x {efficiency:<12.1f}%")

        results.append({
            'threads': num_threads,
            'time': elapsed,
            'speedup': speedup,
            'efficiency': efficiency
        })

    print("=" * 70)

    # Analysis
    print("\nScaling Analysis:")
    print("-" * 70)

    # Find best efficiency and speedup
    best_efficiency = max(results, key=lambda x: x['efficiency'])
    best_speedup = max(results, key=lambda x: x['speedup'])

    print(f"Best efficiency: {best_efficiency['efficiency']:.1f}% at {best_efficiency['threads']} threads")
    print(f"Best speedup: {best_speedup['speedup']:.2f}x at {best_speedup['threads']} threads")

    # Check if scaling is linear
    if len(results) >= 3:
        # Compare speedup at different points
        mid_idx = len(results) // 2
        mid_result = results[mid_idx]

        ideal_speedup = mid_result['threads']
        actual_speedup = mid_result['speedup']
        scaling_factor = actual_speedup / ideal_speedup

        print(f"\nAt {mid_result['threads']} threads:")
        print(f"  Ideal speedup (linear): {ideal_speedup:.1f}x")
        print(f"  Actual speedup: {actual_speedup:.2f}x")
        print(f"  Scaling efficiency: {scaling_factor * 100:.1f}%")

        if scaling_factor > 0.9:
            print("  → Excellent scaling! Near-linear speedup.")
        elif scaling_factor > 0.7:
            print("  → Good scaling. Some overhead from parallelization.")
        elif scaling_factor > 0.5:
            print("  → Moderate scaling. Cache/memory bandwidth limited.")
        else:
            print("  → Poor scaling. Significant bottlenecks.")

    # Extrapolate with best thread count
    print(f"\n{'=' * 70}")
    print("EXTRAPOLATION TO FULL DATASET (1M × 50k)")
    print("=" * 70)

    best_time = min(r['time'] for r in results)
    best_threads = [r for r in results if r['time'] == best_time][0]['threads']

    M_full = 1_000_000
    N_full = 50_000

    pairs_current = N * (N + 1) // 2
    pairs_full = N_full * (N_full + 1) // 2

    scale_factor = (pairs_full / pairs_current) * (M_full / M)
    estimated_time = best_time * scale_factor

    print(f"Best configuration: {best_threads} threads, {best_time:.3f} seconds")
    print(f"Scale factor: {scale_factor:.1f}x")
    print(f"Estimated time: {estimated_time:.1f} seconds")
    print(f"                {estimated_time/3600:.2f} hours")
    print(f"                {estimated_time/86400:.2f} days")
