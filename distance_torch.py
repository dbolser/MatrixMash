#!/usr/bin/env python3
"""PyTorch CUDA implementation for distance matrix computation."""

import numpy as np
import torch
import time

def compute_distance_torch(genotypes: np.ndarray, scores: np.ndarray,
                          chunk_size: int = 5000, device='cuda') -> np.ndarray:
    """
    PyTorch GPU implementation with chunked processing.

    Args:
        genotypes: M × N uint8 array
        scores: M float32 array
        chunk_size: Chunk size for individuals
        device: 'cuda' or 'cpu'

    Returns:
        N × N float32 distance matrix
    """
    M, N = genotypes.shape

    # Transfer scores to GPU once
    scores_gpu = torch.from_numpy(scores).to(device)
    s2_gpu = scores_gpu ** 2  # M

    # Allocate result matrix on CPU
    distances = np.zeros((N, N), dtype=np.float32)

    # Process in chunks
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        n_i = i_end - i_start

        # Transfer genotypes for chunk i
        geno_i = torch.from_numpy(genotypes[:, i_start:i_end]).to(device)  # M × n_i

        for j_start in range(i_start, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            n_j = j_end - j_start

            # Transfer genotypes for chunk j
            geno_j = torch.from_numpy(genotypes[:, j_start:j_end]).to(device)  # M × n_j

            # Allocate output on GPU
            chunk_dist = torch.zeros((n_i, n_j), dtype=torch.float32, device=device)

            # Process in smaller sub-chunks to avoid OOM
            # This reduces peak memory by computing row-by-row instead of full 3D tensor
            # Adaptive sub-chunk based on M: aim for ~500MB intermediate tensors
            # Memory = M × sub_chunk × n_j × 2 bytes (int16)
            target_memory = 500_000_000  # 500 MB
            max_sub_chunk = max(1, target_memory // (M * n_j * 2))
            sub_chunk = min(max_sub_chunk, n_i)
            for ii in range(0, n_i, sub_chunk):
                ii_end = min(ii + sub_chunk, n_i)

                # Get sub-chunk of geno_i: M × sub_n_i
                geno_i_sub = geno_i[:, ii:ii_end]

                # Compute pairwise differences: M × sub_n_i × n_j
                diff = torch.abs(
                    geno_i_sub.unsqueeze(2).to(torch.int16) - geno_j.unsqueeze(1).to(torch.int16)
                )

                # Mask for unknown genotypes
                mask = (geno_i_sub.unsqueeze(2) != 9) & (geno_j.unsqueeze(1) != 9)

                # Compute distances
                s2_broadcast = s2_gpu.unsqueeze(1).unsqueeze(2)
                dist = torch.where(diff == 1, s2_broadcast,
                                 torch.where(diff == 2, 4.0 * s2_broadcast, 0.0))
                dist = torch.where(mask, dist, 0.0)

                # Sum over genotypes and accumulate
                chunk_dist[ii:ii_end, :] = torch.sum(dist, dim=0)

                # Clear intermediate tensors immediately
                del diff, mask, dist

            # Copy back to CPU and store
            distances[i_start:i_end, j_start:j_end] = chunk_dist.cpu().numpy()

            # Mirror to lower triangle if not diagonal block
            if i_start != j_start:
                distances[j_start:j_end, i_start:i_end] = chunk_dist.T.cpu().numpy()

            # Clear GPU memory
            del chunk_dist, geno_j
            torch.cuda.empty_cache()

        del geno_i

    return distances


if __name__ == "__main__":
    from benchmark_utils import (
        create_argument_parser,
        load_data,
        run_standard_benchmark,
        save_output
    )

    parser = create_argument_parser("PyTorch GPU implementation")
    parser.add_argument("--chunk-size", type=int, default=5000,
                       help="Chunk size for processing (default: 5000)")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    genotypes, scores = load_data(args.genotypes_file, args.scores_file)

    # Create wrapper with device and chunk_size
    def compute_with_args(genotypes, scores):
        return compute_distance_torch(genotypes, scores, args.chunk_size, args.device)

    elapsed, distances = run_standard_benchmark(
        compute_with_args,
        f"PyTorch ({args.device.upper()}, chunk_size={args.chunk_size})",
        genotypes,
        scores
    )

    if args.output:
        save_output(args.output, distances)
