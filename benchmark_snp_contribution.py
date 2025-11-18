#!/usr/bin/env python3
"""
Benchmark driver for the Phase 1 SNP contribution CUDA kernel.

This script measures the throughput of `compute_snp_contributions_cuda`
by streaming batches of SNPs from host memory into the GPU, calling the
kernel, and reporting summary statistics (best/median/mean runtimes plus
effective SNP/s and output bandwidth).

Example:
    uv run benchmark_snp_contribution.py genotypes.npy scores.npy --batch-size 32
"""

from __future__ import annotations

import argparse
import ctypes
import math
import statistics
import time
from typing import List, Tuple

import numpy as np

from compute_contributions import (
    Dataset,
    _bind_cuda,
    _load_cuda_lib,
    batch_iterator,
    estimate_safe_batch_size,
    load_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the compute_snp_contributions_cuda entry-point"
    )
    parser.add_argument("genotypes_file", help="Path to the genotypes .npy file")
    parser.add_argument("scores_file", help="Path to the scores .npy file")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of SNPs to process per kernel launch (default: auto via --gpu-mem-gb)",
    )
    parser.add_argument(
        "--gpu-mem-gb",
        type=int,
        default=8,
        help="Approximate GPU memory budget used to pick a safe batch size when --batch-size is omitted",
    )
    parser.add_argument(
        "--max-snps",
        type=int,
        default=None,
        help="Optionally cap the dataset to the first N SNPs for quicker experimentation",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of un-timed passes over the dataset before measuring",
    )
    parser.add_argument(
        "--measurements",
        type=int,
        default=3,
        help="Number of timed passes over the dataset",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-pass timings instead of just the summary",
    )
    return parser.parse_args()


def maybe_trim_dataset(dataset: Dataset, max_snps: int | None) -> Dataset:
    if max_snps is None or max_snps >= dataset.total_snps:
        return dataset

    trimmed_genotypes = np.ascontiguousarray(dataset.genotypes[:max_snps], dtype=np.uint8)
    trimmed_scores = np.ascontiguousarray(dataset.scores[:max_snps], dtype=np.float32)
    return Dataset(
        genotypes=trimmed_genotypes,
        scores=trimmed_scores,
        total_snps=max_snps,
        n_individuals=dataset.n_individuals,
    )


def _run_single_pass(
    dataset: Dataset,
    batch_size: int,
    compute_fn,
    workspace: np.ndarray,
) -> Tuple[float, int]:
    start_time = time.perf_counter()
    bytes_written = 0

    for _, _, geno_batch, score_batch in batch_iterator(dataset, batch_size):
        k_current = geno_batch.shape[0]
        contrib_view = workspace[:k_current]
        err_code = compute_fn(
            geno_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            score_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            contrib_view.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(k_current),
            ctypes.c_int(dataset.n_individuals),
        )
        if err_code != 0:
            raise RuntimeError(
                f"compute_snp_contributions_cuda failed with CUDA error code {err_code}"
            )
        bytes_written += (
            k_current * dataset.n_individuals * dataset.n_individuals * workspace.dtype.itemsize
        )

    elapsed = time.perf_counter() - start_time
    return elapsed, bytes_written


def benchmark(dataset: Dataset, batch_size: int, warmup: int, measurements: int, verbose: bool) -> Tuple[List[float], int]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    batch_size = min(batch_size, dataset.total_snps)

    cuda_lib = _load_cuda_lib()
    compute_fn = _bind_cuda(cuda_lib)

    workspace = np.empty(
        (batch_size, dataset.n_individuals, dataset.n_individuals), dtype=np.float32
    )

    print(f"\nDataset: {dataset.total_snps:,} SNPs × {dataset.n_individuals:,} individuals")
    print(f"Batch size: {batch_size} SNPs ({math.ceil(dataset.total_snps / batch_size)} total batches)")

    for idx in range(warmup):
        print(f"Warmup pass {idx + 1}/{warmup} ...", flush=True)
        _run_single_pass(dataset, batch_size, compute_fn, workspace)

    timings: List[float] = []
    bytes_written: int = 0
    for idx in range(measurements):
        elapsed, bytes_written = _run_single_pass(dataset, batch_size, compute_fn, workspace)
        timings.append(elapsed)
        if verbose:
            print(f"Measurement {idx + 1}/{measurements}: {elapsed:.3f}s")

    return timings, bytes_written


def summarize(timings: List[float], bytes_written: int, dataset: Dataset) -> None:
    if not timings:
        print("No timing samples collected.")
        return

    best = min(timings)
    median = statistics.median(timings)
    mean = statistics.fmean(timings)
    stdev = statistics.pstdev(timings) if len(timings) > 1 else 0.0

    snps_per_sec = dataset.total_snps / best
    gb_per_sec = (bytes_written / best) / 1e9 if bytes_written else 0.0

    print("\nBenchmark summary")
    print("-" * 80)
    print(f"Samples: {len(timings)}")
    print(f"Best   : {best:.3f} s")
    print(f"Median : {median:.3f} s")
    print(f"Mean   : {mean:.3f} s")
    print(f"Stdev  : {stdev:.4f} s")
    print(f"SNP/s  : {snps_per_sec:,.0f}")
    print(f"Output : {bytes_written / 1e9:.3f} GB per pass")
    print(f"BW     : {gb_per_sec:.2f} GB/s effective")
    print("-" * 80)


def main() -> None:
    args = parse_args()

    dataset = load_data(args.genotypes_file, args.scores_file)
    dataset = maybe_trim_dataset(dataset, args.max_snps)

    if args.batch_size is None:
        batch_size = estimate_safe_batch_size(dataset.n_individuals, args.gpu_mem_gb)
    else:
        batch_size = args.batch_size

    timings, bytes_written = benchmark(
        dataset=dataset,
        batch_size=batch_size,
        warmup=max(args.warmup, 0),
        measurements=max(args.measurements, 1),
        verbose=args.verbose,
    )
    summarize(timings, bytes_written, dataset)


if __name__ == "__main__":
    main()

