"""Shared utilities for benchmarking distance matrix implementations."""

import numpy as np
import time
import argparse


def create_argument_parser(description="Distance matrix computation"):
    """Create standard argument parser for benchmark scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("genotypes_file", help="Path to genotypes .npy file")
    parser.add_argument("scores_file", help="Path to scores .npy file")
    parser.add_argument("--output", "-o", help="Save output to file")
    return parser


def load_data(genotypes_file, scores_file):
    """Load genotypes and scores from .npy files."""
    print("Loading data...")
    genotypes = np.load(genotypes_file)
    scores = np.load(scores_file)

    M, N = genotypes.shape
    print(f"  Genotypes: {M:,} × {N:,} (M × N)")
    print(f"  Memory: genotypes={genotypes.nbytes/1e6:.1f} MB, scores={scores.nbytes/1e6:.1f} MB")

    return genotypes, scores


def print_benchmark_header(name):
    """Print benchmark header."""
    print("\n" + "="*70)
    print(f"Benchmarking: {name}")
    print("="*70)


def print_results(distances, elapsed):
    """Print benchmark results."""
    print(f"✓ Completed in {elapsed:.3f} seconds")
    print(f"  Result shape: {distances.shape}")
    print(f"  Result range: [{distances.min():.4f}, {distances.max():.4f}]")
    print(f"  Result mean: {distances.mean():.4f}")
    print(f"  Memory: {distances.nbytes / 1e6:.1f} MB")


def extrapolate_to_full_dataset(M, N, elapsed, M_full=1_000_000, N_full=50_000):
    """Extrapolate timing to full dataset."""
    pairs_current = N * (N + 1) // 2
    pairs_full = N_full * (N_full + 1) // 2

    scale_factor = (pairs_full / pairs_current) * (M_full / M)
    estimated_time = elapsed * scale_factor

    print("\n" + "="*70)
    print("EXTRAPOLATION TO FULL DATASET")
    print("="*70)
    print(f"Current: {M:,} genotypes × {N:,} individuals ({pairs_current:,} pairs)")
    print(f"Target:  {M_full:,} genotypes × {N_full:,} individuals ({pairs_full:,} pairs)")
    print(f"Scale factor: {scale_factor:.1f}x")
    print(f"\nEstimated time for full dataset:")
    print(f"  {estimated_time:.1f} seconds")
    print(f"  {estimated_time/60:.1f} minutes")
    print(f"  {estimated_time/3600:.2f} hours")
    if estimated_time > 86400:
        print(f"  {estimated_time/86400:.2f} days")

    # Throughput
    throughput = pairs_current * M / elapsed
    print(f"\nThroughput: {throughput/1e6:.2f} million comparisons/second")

    return estimated_time


def save_output(filename, distances):
    """Save distance matrix to file."""
    print(f"\nSaving to {filename}...")
    np.save(filename, distances)
    print(f"✓ Saved {distances.nbytes / 1e6:.1f} MB")


def run_standard_benchmark(compute_func, func_name, genotypes, scores):
    """
    Standard benchmark workflow.

    Args:
        compute_func: Function that takes (genotypes, scores) and returns distances
        func_name: Name of the function for display
        genotypes: M × N uint8 array
        scores: M float32 array

    Returns:
        elapsed: Time in seconds
        distances: N × N distance matrix
    """
    M, N = genotypes.shape

    print_benchmark_header(func_name)

    start = time.time()
    distances = compute_func(genotypes, scores)
    elapsed = time.time() - start

    print_results(distances, elapsed)
    extrapolate_to_full_dataset(M, N, elapsed)

    return elapsed, distances
