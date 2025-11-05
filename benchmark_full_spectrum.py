#!/usr/bin/env python3
"""
Full spectrum performance comparison - from dumb Python to optimized CUDA.
One-time benchmark to show the complete journey.
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


def benchmark_pure_python(genotypes, scores):
    """
    Level 0: Pure Python with native lists (slowest possible).
    Converting to Python lists to avoid any NumPy speedups.
    """
    M, N = genotypes.shape

    # Convert to pure Python lists
    geno_list = [[int(genotypes[m, n]) for n in range(N)] for m in range(M)]
    score_list = [float(scores[m]) for m in range(M)]

    # Pure Python computation
    distances = [[0.0 for _ in range(N)] for _ in range(N)]

    start = time.time()

    for i in range(N):
        for j in range(i, N):
            dist = 0.0

            for m in range(M):
                geno_i = geno_list[m][i]
                geno_j = geno_list[m][j]

                # Skip unknown
                if geno_i == 9 or geno_j == 9:
                    continue

                # Compute difference
                diff = abs(geno_i - geno_j)

                s = score_list[m]
                if diff == 1:
                    dist += s * s
                elif diff == 2:
                    dist += 4.0 * s * s

            distances[i][j] = dist
            distances[j][i] = dist

    elapsed = time.time() - start

    # Convert back to numpy for consistency
    distances_np = np.array(distances, dtype=np.float32)

    return elapsed, distances_np


def benchmark_python_with_numpy_arrays(genotypes, scores):
    """
    Level 1: Python loops but with NumPy arrays (current 'reference').
    This is what distance_reference.py does.
    """
    from distance_reference import compute_distance_reference

    start = time.time()
    distances = compute_distance_reference(genotypes, scores)
    elapsed = time.time() - start

    return elapsed, distances


def benchmark_vectorized_numpy(genotypes, scores):
    """
    Level 2: Vectorized NumPy (smart broadcasting).
    Uses NumPy's vectorization capabilities.
    """
    M, N = genotypes.shape

    start = time.time()

    # Pre-compute squared scores
    s2 = scores * scores

    # Initialize output
    distances = np.zeros((N, N), dtype=np.float32)

    # Vectorized computation over M dimension
    # For each pair (i, j), vectorize over all M genotypes
    for i in range(N):
        for j in range(i, N):
            # Get genotype columns for i and j
            geno_i = genotypes[:, i]  # Shape: (M,)
            geno_j = genotypes[:, j]  # Shape: (M,)

            # Mask for valid genotypes (not 9)
            valid = (geno_i != 9) & (geno_j != 9)

            # Compute differences
            diff = np.abs(geno_i.astype(np.int32) - geno_j.astype(np.int32))

            # Compute contributions (vectorized)
            contrib = np.zeros(M, dtype=np.float32)
            contrib[valid & (diff == 1)] = s2[valid & (diff == 1)]
            contrib[valid & (diff == 2)] = 4.0 * s2[valid & (diff == 2)]

            # Sum contributions
            distances[i, j] = contrib.sum()
            distances[j, i] = distances[i, j]

    elapsed = time.time() - start

    return elapsed, distances


def benchmark_numba_cpu(genotypes, scores):
    """Level 3: Numba JIT-compiled CPU (parallel)."""
    from distance_numba_cpu import compute_distance_numba_cpu

    M, N = genotypes.shape

    # Warmup JIT
    small_geno = genotypes[:100, :10] if N > 10 else genotypes[:10, :min(3, N)]
    small_scores = scores[:100] if len(scores) > 100 else scores[:10]
    _ = compute_distance_numba_cpu(small_geno, small_scores)

    start = time.time()
    distances = compute_distance_numba_cpu(genotypes, scores)
    elapsed = time.time() - start

    return elapsed, distances


def benchmark_pytorch_gpu(genotypes, scores):
    """Level 4: PyTorch GPU."""
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
        print(f"      (PyTorch GPU failed: {e})")
        return None, None


def benchmark_cuda_cpp(genotypes, scores):
    """Level 5: Custom CUDA C++ kernel (fastest)."""
    try:
        from distance_cuda_wrapper import compute_distance_cuda

        start = time.time()
        distances = compute_distance_cuda(genotypes, scores)
        elapsed = time.time() - start

        return elapsed, distances
    except Exception as e:
        print(f"      (CUDA C++ failed: {e})")
        return None, None


def main():
    # Test on 1k x 500 dataset
    M, N = 1000, 500

    print("=" * 90)
    print("FULL SPECTRUM PERFORMANCE COMPARISON")
    print("=" * 90)
    print(f"\nDataset: M={M:,} genotypes × N={N:,} individuals")
    print(f"Output matrix: {N:,} × {N:,} = {N*N:,} elements")
    print(f"Comparisons: {N*(N+1)//2:,} pairs × {M:,} genotypes = {N*(N+1)//2*M:,} total")
    print()
    print("Running benchmarks... (this may take a while for slow implementations)")
    print("-" * 90)

    # Generate data
    genotypes, scores = generate_test_data(M, N)

    results = {}
    reference_distances = None

    # Level 0: Pure Python
    print("\n🐌 Level 0: Pure Python (lists, no NumPy optimization)...")
    print("   (This will be VERY slow, please be patient...)")
    try:
        elapsed, distances = benchmark_pure_python(genotypes, scores)
        results['Pure Python'] = elapsed
        reference_distances = distances
        print(f"   ✓ Completed in {elapsed:.3f}s")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Level 1: Python + NumPy arrays
    print("\n🐢 Level 1: Python loops + NumPy arrays (current 'reference')...")
    try:
        elapsed, distances = benchmark_python_with_numpy_arrays(genotypes, scores)
        results['Python + NumPy arrays'] = elapsed
        if reference_distances is None:
            reference_distances = distances
        print(f"   ✓ Completed in {elapsed:.3f}s")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Level 2: Vectorized NumPy
    print("\n🏃 Level 2: Vectorized NumPy (smart broadcasting)...")
    try:
        elapsed, distances = benchmark_vectorized_numpy(genotypes, scores)
        results['Vectorized NumPy'] = elapsed
        if reference_distances is None:
            reference_distances = distances

        # Verify correctness
        if reference_distances is not None:
            if np.allclose(distances, reference_distances, rtol=1e-4, atol=1e-5):
                print(f"   ✓ Completed in {elapsed:.3f}s (results verified)")
            else:
                print(f"   ⚠ Completed in {elapsed:.3f}s (WARNING: results differ!)")
        else:
            print(f"   ✓ Completed in {elapsed:.3f}s")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Level 3: Numba CPU
    print("\n🏃‍♂️ Level 3: Numba JIT CPU (parallel, compiled)...")
    try:
        elapsed, distances = benchmark_numba_cpu(genotypes, scores)
        results['Numba CPU'] = elapsed

        # Verify correctness
        if reference_distances is not None:
            if np.allclose(distances, reference_distances, rtol=1e-4, atol=1e-5):
                print(f"   ✓ Completed in {elapsed:.3f}s (results verified)")
            else:
                print(f"   ⚠ Completed in {elapsed:.3f}s (WARNING: results differ!)")
        else:
            print(f"   ✓ Completed in {elapsed:.3f}s")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Level 4: PyTorch GPU
    print("\n🚀 Level 4: PyTorch GPU...")
    try:
        elapsed, distances = benchmark_pytorch_gpu(genotypes, scores)
        if elapsed is not None:
            results['PyTorch GPU'] = elapsed

            # Verify correctness
            if reference_distances is not None:
                if np.allclose(distances, reference_distances, rtol=1e-4, atol=1e-5):
                    print(f"   ✓ Completed in {elapsed:.3f}s (results verified)")
                else:
                    print(f"   ⚠ Completed in {elapsed:.3f}s (WARNING: results differ!)")
            else:
                print(f"   ✓ Completed in {elapsed:.3f}s")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Level 5: CUDA C++
    print("\n⚡ Level 5: Custom CUDA C++ kernel...")
    try:
        elapsed, distances = benchmark_cuda_cpp(genotypes, scores)
        if elapsed is not None:
            results['CUDA C++'] = elapsed

            # Verify correctness
            if reference_distances is not None:
                if np.allclose(distances, reference_distances, rtol=1e-4, atol=1e-5):
                    print(f"   ✓ Completed in {elapsed:.3f}s (results verified)")
                else:
                    print(f"   ⚠ Completed in {elapsed:.3f}s (WARNING: results differ!)")
            else:
                print(f"   ✓ Completed in {elapsed:.3f}s")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY - THE JOURNEY FROM NAIVE TO OPTIMIZED")
    print("=" * 90)

    if results:
        baseline = results.get('Pure Python', max(results.values()))
        pairs = N * (N + 1) // 2

        print(f"\n{'Level':<8} {'Implementation':<30} {'Time':<12} {'Speedup':<12} {'Throughput':<15}")
        print("-" * 90)

        levels = [
            ('Level 0', 'Pure Python'),
            ('Level 1', 'Python + NumPy arrays'),
            ('Level 2', 'Vectorized NumPy'),
            ('Level 3', 'Numba CPU'),
            ('Level 4', 'PyTorch GPU'),
            ('Level 5', 'CUDA C++'),
        ]

        for level_name, impl in levels:
            if impl in results:
                t = results[impl]
                speedup = baseline / t
                throughput = pairs / t / 1e6

                # Add emoji indicators
                if speedup >= 1000:
                    emoji = "⚡⚡⚡"
                elif speedup >= 100:
                    emoji = "⚡⚡"
                elif speedup >= 10:
                    emoji = "⚡"
                elif speedup >= 2:
                    emoji = "🚀"
                else:
                    emoji = ""

                print(f"{level_name:<8} {impl:<30} {t:>10.3f}s  {speedup:>9.1f}x  {throughput:>10.2f} Mpairs/s  {emoji}")

        print("-" * 90)

        # Highlight the winner
        if results:
            fastest_impl = min(results.items(), key=lambda x: x[1])
            slowest_impl = max(results.items(), key=lambda x: x[1])
            total_speedup = slowest_impl[1] / fastest_impl[1]

            print(f"\n🏆 Fastest: {fastest_impl[0]} at {fastest_impl[1]:.3f}s")
            print(f"🐌 Slowest: {slowest_impl[0]} at {slowest_impl[1]:.3f}s")
            print(f"📊 Total improvement: {total_speedup:.1f}x speedup from slowest to fastest!")

            # Time savings
            time_saved = slowest_impl[1] - fastest_impl[1]
            print(f"⏱️  Time saved: {time_saved:.1f}s on this small dataset")

            # Extrapolate time savings to full dataset
            M_full, N_full = 1_000_000, 50_000
            pairs_test = N * (N + 1) // 2
            pairs_full = N_full * (N_full + 1) // 2
            scale_factor = (pairs_full / pairs_test) * (M_full / M)

            time_saved_full = time_saved * scale_factor
            hours_saved = time_saved_full / 3600
            days_saved = hours_saved / 24

            print(f"\n💡 On full dataset (1M × 50k), this optimization saves:")
            print(f"   {hours_saved:.1f} hours = {days_saved:.1f} days of computation time!")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
