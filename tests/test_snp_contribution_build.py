#!/usr/bin/env python3
"""Utility to verify the CUDA SNP contribution kernel compiles."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = "libsnp_contribution.so"


def main() -> int:
    """Build the CUDA SNP contribution shared library if nvcc is available."""
    if shutil.which("nvcc") is None:
        print(
            "nvcc was not found on PATH; skipping CUDA build test. "
            "Install the CUDA toolkit to run this check.",
            file=sys.stderr,
        )
        return 0

    print("nvcc detected. Building", TARGET)
    subprocess.run(["make", TARGET], cwd=REPO_ROOT, check=True)

    output_path = REPO_ROOT / TARGET
    if not output_path.exists():
        raise FileNotFoundError(
            f"Expected {output_path} to be created by nvcc but it was not."
        )

    print(f"✓ Successfully built {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
