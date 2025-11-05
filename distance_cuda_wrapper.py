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
    from benchmark_utils import (
        create_argument_parser,
        load_data,
        run_standard_benchmark,
        save_output
    )

    parser = create_argument_parser("CUDA C++ kernel implementation")
    args = parser.parse_args()

    genotypes, scores = load_data(args.genotypes_file, args.scores_file)
    elapsed, distances = run_standard_benchmark(
        compute_distance_cuda,
        "CUDA C++ Kernel",
        genotypes,
        scores
    )

    if args.output:
        save_output(args.output, distances)
