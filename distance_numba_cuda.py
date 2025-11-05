#!/usr/bin/env python3
"""Numba CUDA implementation for distance matrix computation."""

import numpy as np
from numba import cuda
import math
import time

@cuda.jit
def compute_distance_kernel(genotypes, scores, distances, M, N):
    """
    CUDA kernel to compute distance matrix.

    Each thread computes one element of the distance matrix.
    Grid is 2D, with blockDim and gridDim handling the N×N matrix.
    """
    # Thread coordinates in the output matrix
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Only compute upper triangle (including diagonal)
    if i < N and j < N and i <= j:
        dist_sum = 0.0

        # Loop over all genotypes
        for m in range(M):
            geno_i = genotypes[m, i]
            geno_j = genotypes[m, j]

            # Skip if either is unknown (9)
            if geno_i == 9 or geno_j == 9:
                continue

            # Compute difference (use np.int32 to avoid unsigned underflow)
            diff = abs(np.int32(geno_i) - np.int32(geno_j))

            # Distance contribution
            s = scores[m]
            if diff == 1:
                dist_sum += s * s
            elif diff == 2:
                dist_sum += 4.0 * s * s

        # Store result (symmetric)
        distances[i, j] = dist_sum
        distances[j, i] = dist_sum


def compute_distance_numba_cuda(genotypes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    Numba CUDA implementation.

    Args:
        genotypes: M × N uint8 array
        scores: M float32 array

    Returns:
        N × N float32 distance matrix
    """
    M, N = genotypes.shape

    # Allocate output
    distances = np.zeros((N, N), dtype=np.float32)

    # Transfer to GPU
    genotypes_gpu = cuda.to_device(genotypes)
    scores_gpu = cuda.to_device(scores)
    distances_gpu = cuda.to_device(distances)

    # Configure kernel launch
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(N / threads_per_block[0])
    blocks_per_grid_y = math.ceil(N / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch kernel
    compute_distance_kernel[blocks_per_grid, threads_per_block](
        genotypes_gpu, scores_gpu, distances_gpu, M, N
    )

    # Copy back to host
    distances_gpu.copy_to_host(distances)

    return distances


def compute_distance_numba_cuda_chunked(genotypes: np.ndarray, scores: np.ndarray,
                                        chunk_size: int = 5000) -> np.ndarray:
    """
    Chunked version for large N that won't fit in GPU memory.

    Args:
        genotypes: M × N uint8 array
        scores: M float32 array
        chunk_size: Process distance matrix in chunks of this size

    Returns:
        N × N float32 distance matrix
    """
    M, N = genotypes.shape

    # Allocate output on host
    distances = np.zeros((N, N), dtype=np.float32)

    # Transfer scores to GPU once
    scores_gpu = cuda.to_device(scores)

    # Process in chunks
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        n_i = i_end - i_start

        # Get genotype chunk for rows i_start:i_end
        geno_chunk = genotypes[:, i_start:i_end].copy()  # M × n_i

        for j_start in range(i_start, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            n_j = j_end - j_start

            # Prepare data for this block
            if i_start == j_start:
                # Diagonal block - use same data
                geno_i_gpu = cuda.to_device(geno_chunk)
                geno_j_gpu = geno_i_gpu
                geno_combined = geno_chunk
            else:
                # Off-diagonal block
                geno_j_chunk = genotypes[:, j_start:j_end].copy()
                # Stack them side by side: M × (n_i + n_j)
                geno_combined = np.hstack([geno_chunk, geno_j_chunk])
                geno_combined_gpu = cuda.to_device(geno_combined)

            # Allocate distance block
            dist_block = np.zeros((n_i, n_j), dtype=np.float32)
            dist_block_gpu = cuda.to_device(dist_block)

            # Compute kernel for this block
            if i_start == j_start:
                # Diagonal block
                threads_per_block = (16, 16)
                blocks_per_grid_x = math.ceil(n_i / threads_per_block[0])
                blocks_per_grid_y = math.ceil(n_j / threads_per_block[1])
                blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

                compute_distance_kernel[blocks_per_grid, threads_per_block](
                    geno_i_gpu, scores_gpu, dist_block_gpu, M, n_i
                )
            else:
                # Off-diagonal block - need custom kernel
                @cuda.jit
                def compute_distance_kernel_rect(geno_combined, scores, distances, M, n_i, n_j):
                    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
                    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

                    if i < n_i and j < n_j:
                        dist_sum = 0.0
                        for m in range(M):
                            geno_i = geno_combined[m, i]
                            geno_j = geno_combined[m, n_i + j]

                            if geno_i == 9 or geno_j == 9:
                                continue

                            diff = abs(np.int32(geno_i) - np.int32(geno_j))
                            s = scores[m]
                            if diff == 1:
                                dist_sum += s * s
                            elif diff == 2:
                                dist_sum += 4.0 * s * s

                        distances[i, j] = dist_sum

                threads_per_block = (16, 16)
                blocks_per_grid_x = math.ceil(n_i / threads_per_block[0])
                blocks_per_grid_y = math.ceil(n_j / threads_per_block[1])
                blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

                compute_distance_kernel_rect[blocks_per_grid, threads_per_block](
                    geno_combined_gpu, scores_gpu, dist_block_gpu, M, n_i, n_j
                )

            # Copy result back
            dist_block_gpu.copy_to_host(dist_block)

            # Store in output matrix
            distances[i_start:i_end, j_start:j_end] = dist_block
            if i_start != j_start:
                distances[j_start:j_end, i_start:i_end] = dist_block.T

    return distances


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Numba CUDA distance matrix")
    parser.add_argument("genotypes_file", help="Path to genotypes .npy file")
    parser.add_argument("scores_file", help="Path to scores .npy file")
    parser.add_argument("--chunk-size", type=int, default=5000,
                       help="Chunk size for processing (default: 5000)")

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    genotypes = np.load(args.genotypes_file)
    scores = np.load(args.scores_file)

    M, N = genotypes.shape
    print(f"Genotypes: {M} × {N}")
    print(f"Chunk size: {args.chunk_size}")

    # Benchmark
    print("\n" + "="*60)
    print("Benchmarking: Numba CUDA (chunked)")
    print("="*60)

    start = time.time()
    if N <= args.chunk_size:
        distances = compute_distance_numba_cuda(genotypes, scores)
    else:
        distances = compute_distance_numba_cuda_chunked(genotypes, scores, args.chunk_size)
    elapsed = time.time() - start

    print(f"✓ Completed in {elapsed:.3f} seconds")
    print(f"  Result shape: {distances.shape}")
    print(f"  Result range: [{distances.min():.2f}, {distances.max():.2f}]")
    print(f"  Memory: {distances.nbytes / 1e6:.1f} MB")

    # Extrapolate
    M_full = 1_000_000
    N_full = 50_000

    scale_factor = (N_full / N) ** 2 * (M_full / M)
    estimated_time = elapsed * scale_factor

    print(f"\n" + "="*60)
    print(f"EXTRAPOLATION TO FULL DATASET")
    print(f"="*60)
    print(f"Current: {M} genotypes × {N} individuals")
    print(f"Target:  {M_full:,} genotypes × {N_full:,} individuals")
    print(f"Scale factor: {scale_factor:.1f}x")
    print(f"Estimated time: {estimated_time:.1f} seconds ({estimated_time/3600:.2f} hours)")

    # Throughput
    pairs_computed = N * (N + 1) // 2
    throughput = pairs_computed / elapsed
    print(f"\nThroughput: {throughput:.0f} pairs/second")

    full_pairs = N_full * (N_full + 1) // 2
    print(f"Full dataset: {full_pairs:,} pairs to compute")
