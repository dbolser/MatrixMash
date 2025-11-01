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
            sub_chunk = min(50, n_i)  # Process 50 rows at a time
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
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch distance matrix")
    parser.add_argument("genotypes_file", help="Path to genotypes .npy file")
    parser.add_argument("scores_file", help="Path to scores .npy file")
    parser.add_argument("--chunk-size", type=int, default=5000,
                       help="Chunk size for processing (default: 5000)")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")

    args = parser.parse_args()

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    print("Loading data...")
    genotypes = np.load(args.genotypes_file)
    scores = np.load(args.scores_file)

    M, N = genotypes.shape
    print(f"Genotypes: {M} × {N}")
    print(f"Chunk size: {args.chunk_size}")

    # Benchmark
    print("\n" + "="*60)
    print(f"Benchmarking: PyTorch ({args.device.upper()})")
    print("="*60)

    start = time.time()
    distances = compute_distance_torch(genotypes, scores, args.chunk_size, args.device)
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
    print(f"Current: {M:,} genotypes × {N:,} individuals")
    print(f"Target:  {M_full:,} genotypes × {N_full:,} individuals")
    print(f"Scale factor: {scale_factor:.1f}x")
    print(f"Estimated time: {estimated_time:.1f} seconds ({estimated_time/3600:.2f} hours)")
    print(f"                ({estimated_time/86400:.2f} days)")

    # Throughput
    pairs_computed = N * (N + 1) // 2
    throughput = pairs_computed / elapsed
    print(f"\nThroughput: {throughput:.0f} pairs/second")

    full_pairs = N_full * (N_full + 1) // 2
    print(f"Full dataset: {full_pairs:,} pairs to compute")
    estimated_full_time = full_pairs / throughput
    print(f"Estimated from throughput: {estimated_full_time:.1f} seconds ({estimated_full_time/3600:.2f} hours)")
