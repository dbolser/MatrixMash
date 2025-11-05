#!/usr/bin/env python3
"""Compute genotype distance matrix using various approaches."""

import numpy as np
import time
from typing import Tuple

def compute_distance_cpu(genotypes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    CPU baseline: compute full N×N distance matrix.

    Args:
        genotypes: M × N uint8 array
        scores: M float32 array

    Returns:
        N × N float32 distance matrix
    """
    M, N = genotypes.shape
    distances = np.zeros((N, N), dtype=np.float32)

    # Precompute squared scores
    s2 = scores ** 2

    for i in range(N):
        for j in range(i, N):  # Only compute upper triangle
            geno_i = genotypes[:, i]
            geno_j = genotypes[:, j]

            # Compute absolute difference
            diff = np.abs(geno_i.astype(np.int16) - geno_j.astype(np.int16))

            # Mask out unknown genotypes (9)
            mask = (geno_i != 9) & (geno_j != 9)

            # Distance based on difference:
            # diff = 0 (same): 0
            # diff = 1 (het): s^2
            # diff = 2 (hom): 4*s^2
            dist = np.where(diff == 1, s2, np.where(diff == 2, 4 * s2, 0.0))
            dist = np.where(mask, dist, 0.0)

            total_dist = np.sum(dist)
            distances[i, j] = total_dist
            distances[j, i] = total_dist

    return distances


def compute_distance_cpu_vectorized(genotypes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    Vectorized CPU approach: compute distance matrix more efficiently.

    Args:
        genotypes: M × N uint8 array
        scores: M float32 array

    Returns:
        N × N float32 distance matrix
    """
    M, N = genotypes.shape
    distances = np.zeros((N, N), dtype=np.float32)

    # Precompute squared scores
    s2 = scores ** 2

    # Process in chunks to avoid memory explosion
    chunk_size = 1000

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        for j_start in range(i_start, N, chunk_size):
            j_end = min(j_start + chunk_size, N)

            # Get genotype chunks: M × chunk_i, M × chunk_j
            geno_i = genotypes[:, i_start:i_end]  # M × n_i
            geno_j = genotypes[:, j_start:j_end]  # M × n_j

            # Compute pairwise differences: M × n_i × n_j
            # Broadcasting: (M, n_i, 1) - (M, 1, n_j) -> (M, n_i, n_j)
            diff = np.abs(
                geno_i[:, :, np.newaxis].astype(np.int16) -
                geno_j[:, np.newaxis, :].astype(np.int16)
            )

            # Mask for unknown genotypes
            mask = (geno_i[:, :, np.newaxis] != 9) & (geno_j[:, np.newaxis, :] != 9)

            # Compute distances
            # s2 is (M,), need to broadcast to (M, n_i, n_j)
            s2_broadcast = s2[:, np.newaxis, np.newaxis]
            dist = np.where(diff == 1, s2_broadcast, np.where(diff == 2, 4 * s2_broadcast, 0.0))
            dist = np.where(mask, dist, 0.0)

            # Sum over genotypes (axis 0)
            chunk_dist = np.sum(dist, axis=0)

            # Store in result matrix
            distances[i_start:i_end, j_start:j_end] = chunk_dist
            if i_start != j_start:  # Don't duplicate diagonal blocks
                distances[j_start:j_end, i_start:i_end] = chunk_dist.T

    return distances


def compute_distance_gpu_cupy(genotypes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    GPU approach using CuPy.

    Args:
        genotypes: M × N uint8 array
        scores: M float32 array

    Returns:
        N × N float32 distance matrix
    """
    try:
        import cupy as cp
    except ImportError:
        raise ImportError("CuPy not installed. Install with: pip install cupy-cuda12x")

    M, N = genotypes.shape

    # Transfer scores to GPU
    scores_gpu = cp.asarray(scores, dtype=cp.float32)
    s2_gpu = scores_gpu ** 2

    # Result matrix on GPU
    distances_gpu = cp.zeros((N, N), dtype=cp.float32)

    # Process in chunks to fit in GPU memory
    chunk_size = 2000  # Adjust based on GPU memory

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)

        # Transfer genotype chunk to GPU
        geno_i_gpu = cp.asarray(genotypes[:, i_start:i_end], dtype=cp.uint8)

        for j_start in range(i_start, N, chunk_size):
            j_end = min(j_start + chunk_size, N)

            geno_j_gpu = cp.asarray(genotypes[:, j_start:j_end], dtype=cp.uint8)

            # Compute pairwise differences
            diff = cp.abs(
                geno_i_gpu[:, :, cp.newaxis].astype(cp.int16) -
                geno_j_gpu[:, cp.newaxis, :].astype(cp.int16)
            )

            # Mask for unknown genotypes
            mask = (geno_i_gpu[:, :, cp.newaxis] != 9) & (geno_j_gpu[:, cp.newaxis, :] != 9)

            # Compute distances
            s2_broadcast = s2_gpu[:, cp.newaxis, cp.newaxis]
            dist = cp.where(diff == 1, s2_broadcast, cp.where(diff == 2, 4 * s2_broadcast, 0.0))
            dist = cp.where(mask, dist, 0.0)

            # Sum over genotypes
            chunk_dist = cp.sum(dist, axis=0)

            # Store in result
            distances_gpu[i_start:i_end, j_start:j_end] = chunk_dist
            if i_start != j_start:
                distances_gpu[j_start:j_end, i_start:i_end] = chunk_dist.T

    # Transfer result back to CPU
    return cp.asnumpy(distances_gpu)


def benchmark(func, genotypes, scores, name):
    """Run a benchmark and return timing."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"Genotypes: {genotypes.shape}, Scores: {scores.shape}")
    print(f"{'='*60}")

    start = time.time()
    result = func(genotypes, scores)
    elapsed = time.time() - start

    print(f"✓ Completed in {elapsed:.3f} seconds")
    print(f"  Result shape: {result.shape}")
    print(f"  Result range: [{result.min():.2f}, {result.max():.2f}]")
    print(f"  Memory: {result.nbytes / 1e6:.1f} MB")

    return elapsed, result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark distance matrix computation")
    parser.add_argument("genotypes_file", help="Path to genotypes .npy file")
    parser.add_argument("scores_file", help="Path to scores .npy file")
    parser.add_argument("--method", choices=["cpu", "cpu-vec", "gpu"],
                       default="gpu", help="Method to use")
    parser.add_argument("--max-individuals", type=int,
                       help="Limit number of individuals for testing")

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    genotypes = np.load(args.genotypes_file)
    scores = np.load(args.scores_file)

    if args.max_individuals:
        genotypes = genotypes[:, :args.max_individuals]
        print(f"Limited to {args.max_individuals} individuals")

    # Select method
    methods = {
        "cpu": (compute_distance_cpu, "CPU Naive"),
        "cpu-vec": (compute_distance_cpu_vectorized, "CPU Vectorized"),
        "gpu": (compute_distance_gpu_cupy, "GPU CuPy"),
    }

    func, name = methods[args.method]
    elapsed, result = benchmark(func, genotypes, scores, name)

    # Estimate full dataset
    M, N = genotypes.shape
    M_full = 1_000_000
    N_full = 50_000

    # Scaling is roughly O(N^2 * M)
    scale_factor = (N_full / N) ** 2 * (M_full / M)
    estimated_time = elapsed * scale_factor

    print(f"\n{'='*60}")
    print(f"EXTRAPOLATION TO FULL DATASET")
    print(f"{'='*60}")
    print(f"Current: {M} genotypes × {N} individuals")
    print(f"Target:  {M_full:,} genotypes × {N_full:,} individuals")
    print(f"Scale factor: {scale_factor:.1f}x")
    print(f"Estimated time: {estimated_time:.1f} seconds ({estimated_time/3600:.2f} hours)")
