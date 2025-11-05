"""Simple reference implementation for validation."""
import numpy as np

def compute_distance_reference(genotypes, scores):
    """
    Simple, obviously correct reference implementation.

    Args:
        genotypes: M × N uint8 array
        scores: M float32 array

    Returns:
        N × N float32 distance matrix
    """
    M, N = genotypes.shape
    distances = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(i, N):  # Only upper triangle
            dist = 0.0

            for m in range(M):
                geno_i = genotypes[m, i]
                geno_j = genotypes[m, j]

                # Skip unknown
                if geno_i == 9 or geno_j == 9:
                    continue

                # Compute difference
                diff = abs(int(geno_i) - int(geno_j))

                s = scores[m]
                if diff == 1:
                    dist += s * s
                elif diff == 2:
                    dist += 4.0 * s * s

            distances[i, j] = dist
            distances[j, i] = dist

    return distances


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Benchmark mode with data files
        from benchmark_utils import (
            create_argument_parser,
            load_data,
            run_standard_benchmark,
            save_output
        )

        parser = create_argument_parser("Reference implementation (pure Python)")
        args = parser.parse_args()

        genotypes, scores = load_data(args.genotypes_file, args.scores_file)
        elapsed, distances = run_standard_benchmark(
            compute_distance_reference,
            "Reference (Pure Python)",
            genotypes,
            scores
        )

        if args.output:
            save_output(args.output, distances)
    else:
        # Test mode with tiny example
        np.random.seed(42)
        genotypes = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]], dtype=np.uint8)
        scores = np.array([1.0, 0.5, 2.0], dtype=np.float32)

        print("Genotypes (M × N):")
        print(genotypes)
        print("\nScores:", scores)

        distances = compute_distance_reference(genotypes, scores)

        print("\nDistances:")
        print(distances)

        print("\nManual verification [0,1]:")
        print("  Gen0: geno[0,0]=0, geno[0,1]=1, diff=1, contrib=1.0²=1.0")
        print("  Gen1: geno[1,0]=1, geno[1,1]=0, diff=1, contrib=0.5²=0.25")
        print("  Gen2: geno[2,0]=2, geno[2,1]=1, diff=1, contrib=2.0²=4.0")
        print("  Total: 1.0 + 0.25 + 4.0 = 5.25")
        print(f"  Computed: {distances[0,1]}")
