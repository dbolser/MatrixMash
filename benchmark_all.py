#!/usr/bin/env python3
"""
Quick performance comparison of all implementations.
"""

import numpy as np
import time

def generate_test_data(M, N, seed=42):
    """Generate test dataset."""
    np.random.seed(seed)
    genotypes = np.random.choice([0, 1, 2, 9], size=(M, N), p=[0.70, 0.20, 0.05, 0.05]).astype(np.uint8)
    scores = np.random.gamma(shape=2.0, scale=0.2, size=M).astype(np.float32)
    scores = np.clip(scores, 0.0, 2.0)
    return genotypes, scores


def benchmark_reference(genotypes, scores):
    """Benchmark reference implementation."""
    from distance_reference import compute_distance_reference

    start = time.time()
    distances = compute_distance_reference(genotypes, scores)
    elapsed = time.time() - start

    return elapsed, distances


def benchmark_cpu(genotypes, scores):
    """Benchmark optimized CPU implementation."""
    from distance_optimized import compute_distance_numba_cpu

    M, N = genotypes.shape
    distances = np.zeros((N, N), dtype=np.float32)

    # Warmup JIT
    if N > 10:
        small_geno = genotypes[:100, :10]
        small_scores = scores[:100]
        small_dist = np.zeros((10, 10), dtype=np.float32)
        compute_distance_numba_cpu(small_geno, small_scores, small_dist)

    start = time.time()
    compute_distance_numba_cpu(genotypes, scores, distances)
    elapsed = time.time() - start

    return elapsed, distances


def benchmark_pytorch(genotypes, scores):
    """Benchmark PyTorch GPU implementation."""
    try:
        import torch
        from distance_torch import compute_distance_torch

        if not torch.cuda.is_available():
            return None, None

        # Warmup
        torch.cuda.synchronize()

        start = time.time()
        distances = compute_distance_torch(genotypes, scores, chunk_size=1000, device='cuda')
        torch.cuda.synchronize()
        elapsed = time.time() - start

        return elapsed, distances
    except Exception as e:
        print(f"  PyTorch failed: {e}")
        return None, None


def benchmark_cuda(genotypes, scores):
    """Benchmark CUDA C++ implementation."""
    try:
        from distance_cuda_wrapper import compute_distance_cuda

        start = time.time()
        distances = compute_distance_cuda(genotypes, scores)
        elapsed = time.time() - start

        return elapsed, distances
    except Exception as e:
        print(f"  CUDA C++ failed: {e}")
        return None, None


def main():
    # Test configurations
    test_configs = [
        (1000, 500, "Small"),
        (5000, 1000, "Medium"),
        (10000, 2000, "Large"),
    ]

    print("=" * 80)
    print("PERFORMANCE COMPARISON - All Implementations")
    print("=" * 80)

    for M, N, label in test_configs:
        print(f"\n{label} dataset: M={M:,} genotypes × N={N:,} individuals")
        print(f"Output matrix: {N:,} × {N:,} = {N*N:,} elements")
        print(f"Comparisons: {N*(N+1)//2:,} pairs × {M:,} genotypes = {N*(N+1)//2*M:,} total")
        print("-" * 80)

        # Generate data
        genotypes, scores = generate_test_data(M, N)

        results = {}

        # Benchmark Reference (only for small datasets)
        if M <= 1000:
            print("Reference (naive Python)...", end=" ", flush=True)
            try:
                elapsed, _ = benchmark_reference(genotypes, scores)
                results['Reference'] = elapsed
                print(f"{elapsed:.3f}s")
            except Exception as e:
                print(f"FAILED: {e}")

        # Benchmark CPU
        print("CPU (Numba optimized)...", end=" ", flush=True)
        try:
            elapsed, _ = benchmark_cpu(genotypes, scores)
            results['CPU'] = elapsed
            print(f"{elapsed:.3f}s")
        except Exception as e:
            print(f"FAILED: {e}")

        # Benchmark PyTorch
        print("PyTorch GPU...", end=" ", flush=True)
        try:
            elapsed, dist_pt = benchmark_pytorch(genotypes, scores)
            if elapsed is not None:
                results['PyTorch'] = elapsed
                print(f"{elapsed:.3f}s")
            else:
                print("SKIPPED")
        except Exception as e:
            print(f"FAILED: {e}")

        # Benchmark CUDA
        print("CUDA C++ kernel...", end=" ", flush=True)
        try:
            elapsed, _ = benchmark_cuda(genotypes, scores)
            if elapsed is not None:
                results['CUDA'] = elapsed
                print(f"{elapsed:.3f}s")
            else:
                print("SKIPPED")
        except Exception as e:
            print(f"FAILED: {e}")

        # Summary
        if results:
            print("\n" + "-" * 80)
            print(f"{'Implementation':<20} {'Time (s)':<12} {'Speedup':<12} {'Throughput (Mpairs/s)':<20}")
            print("-" * 80)

            baseline = min(results.values())
            pairs = N * (N + 1) // 2

            for impl in ['Reference', 'CPU', 'PyTorch', 'CUDA']:
                if impl in results:
                    t = results[impl]
                    speedup = baseline / t
                    throughput = pairs / t / 1e6
                    print(f"{impl:<20} {t:<12.3f} {speedup:<12.2f}x {throughput:<20.2f}")

            print("-" * 80)

    print("\n" + "=" * 80)
    print("EXTRAPOLATION TO FULL DATASET (1M genotypes × 50k individuals)")
    print("=" * 80)

    # Use the largest test for extrapolation
    M_test, N_test = 10000, 2000
    genotypes, scores = generate_test_data(M_test, N_test)

    M_full = 1_000_000
    N_full = 50_000

    pairs_test = N_test * (N_test + 1) // 2
    pairs_full = N_full * (N_full + 1) // 2
    scale_factor = (pairs_full / pairs_test) * (M_full / M_test)

    print(f"\nScale factor from test to full: {scale_factor:.1f}x")
    print(f"Test dataset: {M_test:,} × {N_test:,}")
    print(f"Full dataset: {M_full:,} × {N_full:,}")
    print()

    implementations = [
        ('CPU', benchmark_cpu),
        ('PyTorch', benchmark_pytorch),
        ('CUDA', benchmark_cuda)
    ]

    print(f"{'Implementation':<15} {'Test Time':<12} {'Estimated Full':<20} {'Estimated (hours)':<20}")
    print("-" * 80)

    for name, bench_func in implementations:
        try:
            elapsed, _ = bench_func(genotypes, scores)
            if elapsed is not None:
                estimated = elapsed * scale_factor
                hours = estimated / 3600
                print(f"{name:<15} {elapsed:<12.3f}s {estimated:<20.1f}s {hours:<20.2f}h")
        except Exception as e:
            print(f"{name:<15} FAILED: {e}")

    print("=" * 80)


if __name__ == "__main__":
    main()
