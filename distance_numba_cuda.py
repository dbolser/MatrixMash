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

            # Compute absolute difference (avoid unsigned underflow)
            # Manual absolute difference to ensure correct signed arithmetic
            if geno_i > geno_j:
                diff = geno_i - geno_j
            else:
                diff = geno_j - geno_i

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
    import sys

    print("="*70)
    print("⚠ WARNING: Numba CUDA implementation is currently unstable")
    print("="*70)
    print("This implementation has known issues and may crash.")
    print("For production use, please use instead:")
    print("  - distance_cuda_wrapper.py (CUDA C++ - fastest, most stable)")
    print("  - distance_torch.py (PyTorch GPU - fast, stable)")
    print("  - distance_numba_cpu.py (Numba CPU - stable)")
    print("="*70)
    print()

    response = input("Continue anyway? (y/N): ").strip().lower()
    if response != 'y':
        print("Exiting.")
        sys.exit(0)

    from benchmark_utils import (
        create_argument_parser,
        load_data,
        run_standard_benchmark,
        save_output
    )

    parser = create_argument_parser("Numba CUDA GPU implementation (EXPERIMENTAL)")
    args = parser.parse_args()

    try:
        genotypes, scores = load_data(args.genotypes_file, args.scores_file)
        elapsed, distances = run_standard_benchmark(
            compute_distance_numba_cuda,
            "Numba CUDA GPU (EXPERIMENTAL)",
            genotypes,
            scores
        )

        if args.output:
            save_output(args.output, distances)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nThis implementation is experimental and unstable.")
        print("Please use distance_cuda_wrapper.py or distance_torch.py instead.")
        sys.exit(1)
