# MatrixMash

## Testing the SNP Contribution Pipeline

Run the Python bytecode compilation check to ensure the wrapper has no syntax errors:

```bash
python -m compileall compute_contributions.py
```

To make sure the CUDA shared library builds, run the helper test script. It will call
`make libsnp_contribution.so` whenever `nvcc` is on your PATH and otherwise skip with an
informative message.

```bash
uv run tests/test_snp_contribution_build.py
```

### Installing `nvcc` without a GPU

The build helper only compiles the CUDA shared object; it never executes GPU kernels, so
you can run it on machines that do not have an NVIDIA GPU.  Install the CUDA Toolkit for
your platform (for example, on Ubuntu you can use `sudo apt install cuda-toolkit-12-4`),
ensure that `<cuda>/bin` is on your `PATH`, and `nvcc --version` should succeed.  Even on
CPU-only hosts this is enough for `tests/test_snp_contribution_build.py` to verify that the
kernel code still compiles.  You only need real GPU hardware when you want to _run_ the
pipeline, not when you are just checking that the code builds.
