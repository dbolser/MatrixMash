#!/usr/bin/env python3
"""
Optimized CPU implementation using numba JIT.
This will give us a baseline for scaling experiments.
"""

import numpy as np
from numba import njit, prange
import time

@njit(parallel=True, fastmath=True)
def compute_distance_numba_cpu(genotypes, scores, distances):
    """
    Numba JIT-compiled CPU implementation with parallel loops.

    Args:
        genotypes: M × N uint8 array
        scores: M float32 array
        distances: N × N float32 array (output, pre-allocated)
    """
    M, N = genotypes.shape

    # Precompute squared scores
    s2 = scores * scores

    # Parallel outer loop over pairs
    for i in prange(N):
        for j in range(i, N):
            dist_sum = 0.0

            # Inner loop over genotypes
            for m in range(M):
                geno_i = genotypes[m, i]
                geno_j = genotypes[m, j]

                # Skip unknown genotypes
                if geno_i == 9 or geno_j == 9:
                    continue

                # Compute difference
                diff = abs(int(geno_i) - int(geno_j))

                # Add distance contribution
                if diff == 1:
                    dist_sum += s2[m]
                elif diff == 2:
                    dist_sum += 4.0 * s2[m]

            # Store symmetric result
            distances[i, j] = dist_sum
            distances[j, i] = dist_sum


def run_benchmark(genotypes, scores, name="Numba CPU"):
    """Run benchmark and return timing."""
    M, N = genotypes.shape
    distances = np.zeros((N, N), dtype=np.float32)

    print("\n" + "="*60)
    print(f"Benchmarking: {name}")
    print(f"Genotypes: {M:,} × {N:,}")
    print("="*60)

    # Warmup JIT compilation
    if N > 10:
        print("Warming up JIT compilation...")
        small_geno = genotypes[:100, :10]
        small_scores = scores[:100]
        small_dist = np.zeros((10, 10), dtype=np.float32)
        compute_distance_numba_cpu(small_geno, small_scores, small_dist)

    # Actual benchmark
    start = time.time()
    compute_distance_numba_cpu(genotypes, scores, distances)
    elapsed = time.time() - start

    print(f"✓ Completed in {elapsed:.3f} seconds")
    print(f"  Result shape: {distances.shape}")
    print(f"  Result range: [{distances.min():.2f}, {distances.max():.2f}]")
    print(f"  Memory: {distances.nbytes / 1e6:.1f} MB")

    return elapsed, distances


def extrapolate_timing(M_current, N_current, elapsed, M_full=1_000_000, N_full=50_000):
    """Extrapolate timing to full dataset."""
    # Complexity is O(N^2 * M) for upper triangle computation
    # Since we only compute upper triangle, it's actually N*(N+1)/2 * M
    pairs_current = N_current * (N_current + 1) // 2
    pairs_full = N_full * (N_full + 1) // 2

    scale_factor_pairs = pairs_full / pairs_current
    scale_factor_M = M_full / M_current
    scale_factor = scale_factor_pairs * scale_factor_M

    estimated_time = elapsed * scale_factor

    print(f"\n" + "="*60)
    print(f"EXTRAPOLATION TO FULL DATASET")
    print(f"="*60)
    print(f"Current: {M_current:,} genotypes × {N_current:,} individuals")
    print(f"         {pairs_current:,} pairs")
    print(f"Target:  {M_full:,} genotypes × {N_full:,} individuals")
    print(f"         {pairs_full:,} pairs")
    print(f"")
    print(f"Scale factor: {scale_factor:.1f}x")
    print(f"  From pairs: {scale_factor_pairs:.1f}x")
    print(f"  From genotypes: {scale_factor_M:.1f}x")
    print(f"")
    print(f"Estimated time: {estimated_time:.1f} seconds")
    print(f"                {estimated_time/60:.1f} minutes")
    print(f"                {estimated_time/3600:.2f} hours")
    if estimated_time > 86400:
        print(f"                {estimated_time/86400:.2f} days")

    # Throughput analysis
    throughput = pairs_current * M_current / elapsed
    print(f"\nThroughput: {throughput:.0f} pair-genotype comparisons/second")
    print(f"            {throughput/1e6:.1f} million comparisons/second")

    total_comparisons = pairs_full * M_full
    print(f"Full dataset requires: {total_comparisons:.2e} comparisons")

    return estimated_time


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimized CPU distance matrix")
    parser.add_argument("genotypes_file", help="Path to genotypes .npy file")
    parser.add_argument("scores_file", help="Path to scores .npy file")
    parser.add_argument("--save", help="Save output to file")

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    genotypes = np.load(args.genotypes_file)
    scores = np.load(args.scores_file)

    M, N = genotypes.shape

    # Run benchmark
    elapsed, distances = run_benchmark(genotypes, scores)

    # Extrapolate
    extrapolate_timing(M, N, elapsed)

    # Save if requested
    if args.save:
        print(f"\nSaving to {args.save}...")
        np.save(args.save, distances)
        print(f"Saved {distances.nbytes / 1e6:.1f} MB")
