#!/usr/bin/env python3
"""Phase 1 pipeline for the Compute-Once, Assemble-Many strategy."""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np

LIB_NAME = "libsnp_contribution.so"

# Module-level constants for dtype sizes (avoid repeated dtype instantiation)
FLOAT32_SIZE = np.dtype(np.float32).itemsize
UINT8_SIZE = np.dtype(np.uint8).itemsize

ComputeFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
)


def _load_cuda_lib() -> ctypes.CDLL:
    lib_path = os.path.join(os.path.dirname(__file__), LIB_NAME)
    try:
        return ctypes.CDLL(lib_path)
    except OSError as exc:  # pragma: no cover - informative error path
        raise ImportError(
            f"Failed to load CUDA library '{LIB_NAME}' from {lib_path}.\n"
            "Did you run 'make libsnp_contribution.so'?"
        ) from exc


def _bind_cuda(lib: ctypes.CDLL) -> ComputeFn:
    return ComputeFn(("compute_snp_contributions_cuda", lib))


@dataclass
class Dataset:
    genotypes: np.ndarray  # shape (M, N) uint8
    scores: np.ndarray  # shape (M,) float32
    total_snps: int
    n_individuals: int


def load_data(genotypes_file: str, scores_file: str) -> Dataset:
    genotypes = np.load(genotypes_file)
    scores = np.load(scores_file)

    if genotypes.ndim != 2:
        raise ValueError("Genotypes array must be 2-D")
    if scores.ndim != 1:
        raise ValueError("Scores array must be 1-D")
    if genotypes.shape[0] != scores.shape[0]:
        raise ValueError("Genotypes and scores must share the same SNP dimension")

    genotypes = np.ascontiguousarray(genotypes, dtype=np.uint8)
    scores = np.ascontiguousarray(scores, dtype=np.float32)

    m, n = genotypes.shape
    print(f"Loaded {m:,} SNPs across {n:,} individuals")
    return Dataset(genotypes, scores, m, n)


def estimate_safe_batch_size(n: int, gpu_memory_gb: int) -> int:
    """Estimate a safe SNP batch size that fits in GPU memory."""

    usable_bytes = gpu_memory_gb * (1024**3) * 0.80
    bytes_per_snp = (
        (n * n * FLOAT32_SIZE)
        + (n * UINT8_SIZE)
        + FLOAT32_SIZE
    )  # output + genotypes + score
    safe_k = int(usable_bytes // bytes_per_snp)
    safe_k = max(1, min(safe_k, 4096))

    print(f"GPU memory budget: {gpu_memory_gb} GB")
    print(f"  Approx bytes per SNP contribution matrix: {bytes_per_snp / 1e6:.2f} MB")
    print(f"  Using SNP batch size k={safe_k}")
    return safe_k


def batch_iterator(dataset: Dataset, batch_size: int):
    for start in range(0, dataset.total_snps, batch_size):
        end = min(start + batch_size, dataset.total_snps)
        yield start, end, dataset.genotypes[start:end], dataset.scores[start:end]


def run_phase1(dataset: Dataset, batch_size: int, compute_fn: ComputeFn) -> None:
    for start, end, geno_batch, score_batch in batch_iterator(dataset, batch_size):
        k_current = end - start

        print("=" * 80)
        print(f"Processing SNP batch [{start}, {end}) (k={k_current})")
        print("=" * 80)

        contributions = np.zeros((k_current, dataset.n_individuals, dataset.n_individuals), dtype=np.float32)

        start_time = time.time()
        err_code = compute_fn(
            geno_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            score_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            contributions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(k_current),
            ctypes.c_int(dataset.n_individuals),
        )
        if err_code != 0:
            raise RuntimeError(
                f"compute_snp_contributions_cuda failed with CUDA error code {err_code}"
            )
        elapsed = time.time() - start_time

        print(f"Phase 1 kernel finished in {elapsed:.3f}s")
        contrib_min = float(contributions.min()) if contributions.size else 0.0
        contrib_max = float(contributions.max()) if contributions.size else 0.0
        print(
            f"  Output tensor: shape={contributions.shape} size={contributions.nbytes / 1e6:.2f} MB "
            f"min={contrib_min:.3f} max={contrib_max:.3f}"
        )

        # Placeholder for Phase 2 hook
        # assemble_terms(contributions)


def parse_args(argv: Tuple[str, ...]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 SNP contribution pipeline")
    parser.add_argument("genotypes_file", help="Path to the genotypes .npy file")
    parser.add_argument("scores_file", help="Path to the scores .npy file")
    parser.add_argument(
        "--gpu-mem-gb",
        type=int,
        default=8,
        help="Approximate GPU memory budget in GB (default: 8)",
    )
    return parser.parse_args(argv)


def main(argv: Tuple[str, ...] | None = None) -> None:
    args = parse_args(argv or tuple(sys.argv[1:]))
    dataset = load_data(args.genotypes_file, args.scores_file)
    batch_size = estimate_safe_batch_size(dataset.n_individuals, args.gpu_mem_gb)
    cuda_lib = _load_cuda_lib()
    compute_fn = _bind_cuda(cuda_lib)
    run_phase1(dataset, batch_size, compute_fn)
    print("Phase 1 pipeline complete")


if __name__ == "__main__":
    main()
