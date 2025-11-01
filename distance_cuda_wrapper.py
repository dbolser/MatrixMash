#!/usr/bin/env python3
"""
Python wrapper for CUDA distance matrix computation.
Uses ctypes to call the compiled CUDA library.
"""

import numpy as np
import ctypes
import os
import time

# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), 'libdistance_cuda.so')

try:
    _cuda_lib = ctypes.CDLL(lib_path)
except OSError as e:
    raise ImportError(f"Failed to load CUDA library from {lib_path}. "
                     f"Did you compile it? Run: make distance_cuda\n"
                     f"Error: {e}")

# Define function signature
_cuda_lib.compute_distance_cuda.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # genotypes
    ctypes.POINTER(ctypes.c_float),   # scores
    ctypes.POINTER(ctypes.c_float),   # distances (output)
    ctypes.c_int,                     # M
    ctypes.c_int                      # N
]
_cuda_lib.compute_distance_cuda.restype = None


def compute_distance_cuda(genotypes, scores):
    """
    Compute distance matrix using CUDA kernel.

    Args:
        genotypes: M × N uint8 numpy array
        scores: M float32 numpy array

    Returns:
        N × N float32 numpy array (distance matrix)
    """
    # Validate inputs
    if not isinstance(genotypes, np.ndarray):
        genotypes = np.array(genotypes)
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)

    if genotypes.dtype != np.uint8:
        genotypes = genotypes.astype(np.uint8)
    if scores.dtype != np.float32:
        scores = scores.astype(np.float32)

    M, N = genotypes.shape

    # Ensure C-contiguous memory layout
    genotypes = np.ascontiguousarray(genotypes)
    scores = np.ascontiguousarray(scores)

    # Allocate output
    distances = np.zeros((N, N), dtype=np.float32)

    # Get pointers
    genotypes_ptr = genotypes.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    scores_ptr = scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    distances_ptr = distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call CUDA kernel
    _cuda_lib.compute_distance_cuda(
        genotypes_ptr,
        scores_ptr,
        distances_ptr,
        M,
        N
    )

    return distances


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CUDA distance matrix computation")
    parser.add_argument("genotypes_file", help="Path to genotypes .npy file")
    parser.add_argument("scores_file", help="Path to scores .npy file")
    parser.add_argument("--output", "-o", help="Save output to file")

    args = parser.parse_args()

    print("Loading data...")
    genotypes = np.load(args.genotypes_file)
    scores = np.load(args.scores_file)

    M, N = genotypes.shape
    print(f"Genotypes: {M:,} × {N:,}")
    print(f"Input size: {genotypes.nbytes/1e6:.1f} MB")

    print("\n" + "="*60)
    print("Benchmarking: CUDA C++ Kernel")
    print("="*60)

    start = time.time()
    distances = compute_distance_cuda(genotypes, scores)
    elapsed = time.time() - start

    print(f"✓ Completed in {elapsed:.3f} seconds")
    print(f"  Result shape: {distances.shape}")
    print(f"  Result range: [{distances.min():.2f}, {distances.max():.2f}]")
    print(f"  Memory: {distances.nbytes / 1e6:.1f} MB")

    # Extrapolate
    M_full = 1_000_000
    N_full = 50_000

    pairs_computed = N * (N + 1) // 2
    throughput = pairs_computed / elapsed
    comparisons_per_sec = pairs_computed * M / elapsed

    print(f"\nThroughput: {throughput:.0f} pairs/second")
    print(f"Comparisons: {comparisons_per_sec/1e6:.1f} M/sec")

    scale_factor = (N_full / N) ** 2 * (M_full / M)
    estimated_time = elapsed * scale_factor

    print(f"\n" + "="*60)
    print(f"EXTRAPOLATION TO FULL DATASET")
    print(f"="*60)
    print(f"Current: {M:,} genotypes × {N:,} individuals")
    print(f"Target:  {M_full:,} genotypes × {N_full:,} individuals")
    print(f"Scale factor: {scale_factor:.1f}x")
    print(f"Estimated time: {estimated_time:.1f} seconds ({estimated_time/3600:.2f} hours)")
    print(f"                ({estimated_time/86400:.2f} days)")

    full_pairs = N_full * (N_full + 1) // 2
    estimated_full_time = full_pairs * M_full / comparisons_per_sec
    print(f"Estimated from throughput: {estimated_full_time:.1f} seconds ({estimated_full_time/3600:.2f} hours)")

    # Save if requested
    if args.output:
        print(f"\nSaving to {args.output}...")
        np.save(args.output, distances)
        print(f"Saved {distances.nbytes / 1e6:.1f} MB")
