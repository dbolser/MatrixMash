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
