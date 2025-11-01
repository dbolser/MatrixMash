#!/usr/bin/env python3
"""Generate synthetic genotype data for benchmarking."""

import numpy as np
import argparse

def generate_genotype_data(n_genotypes, n_individuals, seed=42):
    """
    Generate synthetic genotype matrix and scores.

    Genotypes follow realistic biological distributions:
    - Most are homozygous reference (0) or unknown (9)
    - Some heterozygous (1)
    - Fewer homozygous alternate (2)

    Args:
        n_genotypes: Number of genotypes (M)
        n_individuals: Number of individuals (N)
        seed: Random seed for reproducibility

    Returns:
        genotypes: M x N uint8 array with values in {0, 1, 2, 9}
        scores: M float32 array with consequence scores
    """
    np.random.seed(seed)

    # Realistic genotype distribution
    # ~70% ref/ref, ~20% het, ~5% alt/alt, ~5% unknown
    probabilities = [0.70, 0.20, 0.05, 0.05]

    genotypes = np.random.choice(
        [0, 1, 2, 9],
        size=(n_genotypes, n_individuals),
        p=probabilities
    ).astype(np.uint8)

    # Scores typically range from 0 (benign) to ~1 (damaging)
    # Use gamma distribution to get realistic skew
    scores = np.random.gamma(shape=2.0, scale=0.2, size=n_genotypes).astype(np.float32)
    scores = np.clip(scores, 0.0, 2.0)  # Cap at 2.0

    return genotypes, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic genotype data")
    parser.add_argument("-m", "--genotypes", type=int, default=10000,
                       help="Number of genotypes (default: 10000)")
    parser.add_argument("-n", "--individuals", type=int, default=1000,
                       help="Number of individuals (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("-o", "--output-prefix", type=str, default="genotype_data",
                       help="Output file prefix (default: genotype_data)")

    args = parser.parse_args()

    print(f"Generating {args.genotypes} genotypes × {args.individuals} individuals...")
    genotypes, scores = generate_genotype_data(args.genotypes, args.individuals, args.seed)

    print(f"Genotype matrix: {genotypes.shape}, {genotypes.dtype}, {genotypes.nbytes / 1e6:.1f} MB")
    print(f"Scores: {scores.shape}, {scores.dtype}, {scores.nbytes / 1e6:.1f} MB")

    # Save to files
    np.save(f"{args.output_prefix}_genotypes.npy", genotypes)
    np.save(f"{args.output_prefix}_scores.npy", scores)

    print(f"Saved to {args.output_prefix}_genotypes.npy and {args.output_prefix}_scores.npy")
