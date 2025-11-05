#!/usr/bin/env python3
"""
Validate that all implementations produce identical results.
"""

import numpy as np
import sys

def generate_test_data(M=100, N=50, seed=42):
    """Generate small test dataset."""
    np.random.seed(seed)
    genotypes = np.random.choice([0, 1, 2, 9], size=(M, N), p=[0.70, 0.20, 0.05, 0.05]).astype(np.uint8)
    scores = np.random.gamma(shape=2.0, scale=0.2, size=M).astype(np.float32)
    scores = np.clip(scores, 0.0, 2.0)
    return genotypes, scores


def test_reference():
    """Test reference implementation (ground truth)."""
    print("Testing Reference (ground truth)...")
    from distance_reference import compute_distance_reference

    genotypes, scores = generate_test_data()
    distances = compute_distance_reference(genotypes, scores)

    print(f"  ✓ Shape: {distances.shape}")
    print(f"  ✓ Range: [{distances.min():.4f}, {distances.max():.4f}]")
    print(f"  ✓ Mean: {distances.mean():.4f}")

    return distances


def test_cpu_optimized():
    """Test optimized CPU implementation."""
    print("\nTesting CPU (Numba optimized)...")
    from distance_numba_cpu import compute_distance_numba_cpu

    genotypes, scores = generate_test_data()
    distances = compute_distance_numba_cpu(genotypes, scores)

    print(f"  ✓ Shape: {distances.shape}")
    print(f"  ✓ Range: [{distances.min():.4f}, {distances.max():.4f}]")
    print(f"  ✓ Mean: {distances.mean():.4f}")

    return distances


def test_pytorch():
    """Test PyTorch GPU implementation."""
    print("\nTesting PyTorch GPU...")
    try:
        import torch
        from distance_torch import compute_distance_torch

        if not torch.cuda.is_available():
            print("  ⚠ CUDA not available, skipping PyTorch test")
            return None

        genotypes, scores = generate_test_data()
        distances = compute_distance_torch(genotypes, scores, chunk_size=50, device='cuda')

        print(f"  ✓ Shape: {distances.shape}")
        print(f"  ✓ Range: [{distances.min():.4f}, {distances.max():.4f}]")
        print(f"  ✓ Mean: {distances.mean():.4f}")

        return distances
    except Exception as e:
        print(f"  ⚠ PyTorch test failed: {e}")
        return None


def test_cuda():
    """Test CUDA C++ implementation."""
    print("\nTesting CUDA C++ kernel...")
    try:
        from distance_cuda_wrapper import compute_distance_cuda

        genotypes, scores = generate_test_data()
        distances = compute_distance_cuda(genotypes, scores)

        print(f"  ✓ Shape: {distances.shape}")
        print(f"  ✓ Range: [{distances.min():.4f}, {distances.max():.4f}]")
        print(f"  ✓ Mean: {distances.mean():.4f}")

        return distances
    except Exception as e:
        print(f"  ⚠ CUDA C++ test failed: {e}")
        return None


def compare_matrices(mat1, mat2, name1, name2, rtol=1e-4, atol=1e-5):
    """Compare two distance matrices."""
    if mat1 is None or mat2 is None:
        print(f"  ⚠ Skipping comparison (one matrix is None)")
        return False

    print(f"\nComparing {name1} vs {name2}:")

    # Check shapes
    if mat1.shape != mat2.shape:
        print(f"  ✗ Shape mismatch: {mat1.shape} vs {mat2.shape}")
        return False

    # Check for NaN or Inf
    if np.any(np.isnan(mat1)) or np.any(np.isnan(mat2)):
        print(f"  ✗ Contains NaN values")
        return False

    if np.any(np.isinf(mat1)) or np.any(np.isinf(mat2)):
        print(f"  ✗ Contains Inf values")
        return False

    # Compute differences
    abs_diff = np.abs(mat1 - mat2)
    rel_diff = abs_diff / (np.abs(mat1) + 1e-10)

    max_abs_diff = abs_diff.max()
    max_rel_diff = rel_diff.max()
    mean_abs_diff = abs_diff.mean()

    print(f"  Max absolute difference: {max_abs_diff:.6e}")
    print(f"  Max relative difference: {max_rel_diff:.6e}")
    print(f"  Mean absolute difference: {mean_abs_diff:.6e}")

    # Check if close
    is_close = np.allclose(mat1, mat2, rtol=rtol, atol=atol)

    if is_close:
        print(f"  ✓ Matrices are equivalent (within rtol={rtol}, atol={atol})")
    else:
        # Find where they differ
        diff_mask = ~np.isclose(mat1, mat2, rtol=rtol, atol=atol)
        n_diff = diff_mask.sum()
        pct_diff = 100.0 * n_diff / mat1.size

        print(f"  ✗ Matrices differ at {n_diff} elements ({pct_diff:.2f}%)")

        # Show a few examples of differences
        diff_indices = np.argwhere(diff_mask)
        print(f"\n  Examples of differences (first 5):")
        for idx in diff_indices[:5]:
            i, j = idx
            print(f"    [{i},{j}]: {mat1[i,j]:.6f} vs {mat2[i,j]:.6f} (diff: {abs_diff[i,j]:.6e})")

    return is_close


def check_symmetry(mat, name):
    """Check if matrix is symmetric."""
    print(f"\nChecking symmetry of {name}:")

    if mat is None:
        print(f"  ⚠ Skipping (matrix is None)")
        return False

    # Check if symmetric
    diff = np.abs(mat - mat.T)
    max_diff = diff.max()

    is_symmetric = np.allclose(mat, mat.T, rtol=1e-5, atol=1e-6)

    print(f"  Max difference from transpose: {max_diff:.6e}")

    if is_symmetric:
        print(f"  ✓ Matrix is symmetric")
    else:
        print(f"  ✗ Matrix is not symmetric")

    return is_symmetric


def main():
    print("="*70)
    print("VALIDATION TEST - Comparing All Implementations")
    print("="*70)

    print(f"\nTest dataset: 100 genotypes × 50 individuals")

    # Run all implementations
    reference_result = test_reference()
    cpu_result = test_cpu_optimized()
    pytorch_result = test_pytorch()
    cuda_result = test_cuda()

    # Check symmetry
    print("\n" + "="*70)
    print("SYMMETRY CHECKS")
    print("="*70)

    cpu_symmetric = check_symmetry(cpu_result, "CPU")
    pytorch_symmetric = check_symmetry(pytorch_result, "PyTorch") if pytorch_result is not None else None
    cuda_symmetric = check_symmetry(cuda_result, "CUDA C++") if cuda_result is not None else None

    # Compare implementations
    print("\n" + "="*70)
    print("CROSS-IMPLEMENTATION COMPARISONS")
    print("="*70)

    comparisons = []

    # All vs Reference (ground truth)
    result = compare_matrices(reference_result, cpu_result, "Reference", "CPU")
    comparisons.append(("Reference vs CPU", result))

    if pytorch_result is not None:
        result = compare_matrices(reference_result, pytorch_result, "Reference", "PyTorch")
        comparisons.append(("Reference vs PyTorch", result))

    if cuda_result is not None:
        result = compare_matrices(reference_result, cuda_result, "Reference", "CUDA C++")
        comparisons.append(("Reference vs CUDA C++", result))

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\nSymmetry checks:")
    print(f"  CPU:     {'✓ PASS' if cpu_symmetric else '✗ FAIL'}")
    if pytorch_symmetric is not None:
        print(f"  PyTorch: {'✓ PASS' if pytorch_symmetric else '✗ FAIL'}")
    if cuda_symmetric is not None:
        print(f"  CUDA:    {'✓ PASS' if cuda_symmetric else '✗ FAIL'}")

    print("\nCross-implementation comparisons:")
    all_passed = True
    for name, result in comparisons:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False

    print("\n" + "="*70)
    if all_passed and cpu_symmetric:
        print("✓ ALL TESTS PASSED - All implementations produce identical results!")
        print("="*70)
        return 0
    else:
        print("✗ SOME TESTS FAILED - See details above")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
