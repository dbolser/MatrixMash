# Distance Matrix Computation - Scaling Analysis

## Problem Description

- **Input**: M genotypes × N individuals matrix, values in {0, 1, 2, 9}
- **Goal**: Compute N×N symmetric distance matrix
- **Distance function**:
  - Same homozygous (0↔0, 1↔1, 2↔2): 0
  - Heterozygous (0↔1, 1↔2): s²
  - Homozygous (0↔2): 4s²
  - Unknown (any↔9): 0
  - Sum across all M genotypes

## Target Dataset

- **Genotypes**: 1,000,000
- **Individuals**: 50,000
- **Input size**: ~50 GB (uint8)
- **Output size**: 10 GB (50k × 50k × 4 bytes)
- **Computational complexity**: 1.25 × 10¹⁵ pair-genotype comparisons

## Implementation Approaches

### 1. Optimized CPU (Numba JIT with parallel loops)
**Status**: ✅ Working and benchmarked

Implementation in `distance_optimized.py`

### 2. GPU Implementations
**Status**: ⚠️ Not tested (CUDA toolkit installation issues)

Code ready in:
- `distance_torch.py` (PyTorch)
- `distance_numba.py` (Numba CUDA)
- `distance_matrix.py` (CuPy)

## Scaling Experiments

| Genotypes (M) | Individuals (N) | Time (sec) | Throughput (M cmp/s) | Extrapolated Time |
|--------------|----------------|------------|---------------------|-------------------|
| 10,000       | 500            | 0.96       | 1,301               | 11.1 days         |
| 50,000       | 1,000          | 15.3       | 1,632               | 8.9 days          |
| 100,000      | 2,000          | 275.7      | 726                 | 19.9 days         |

**Key Observations**:
1. Throughput decreases with larger datasets (cache effects)
2. Larger tests give more conservative estimates
3. Most realistic estimate from largest test: **~20 days on CPU**

## Estimated Timings for Full Dataset (1M × 50k)

### CPU (64-core system)
- **Optimized Numba**: 9-20 days
  - Conservative estimate: **~20 days**
  - Optimistic estimate: ~9 days
- **Naive Python**: 128+ days (not practical)

### GPU (8 GB RTX 4060)
- **Expected**: 0.5-2 days (10-50x speedup over CPU)
- **Status**: Not tested due to CUDA setup issues

## Recommendations

### 1. For Immediate Use: CPU Implementation

Use `distance_optimized.py` - it's ready to go:

```bash
python distance_optimized.py genotypes.npy scores.npy --save output.npy
```

**Pros**:
- Works now, no setup needed
- ~20 days for full dataset
- Reliable and simple

**Cons**:
- Slow for 1M × 50k dataset

### 2. For Production: GPU Implementation

Fix CUDA setup and use one of the GPU implementations:

**Option A: PyTorch** (recommended)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
python distance_torch.py genotypes.npy scores.npy --chunk-size 5000
```

**Option B: Numba CUDA**
```bash
conda install cudatoolkit
python distance_numba.py genotypes.npy scores.npy
```

**Expected speedup**: 10-50x → **0.5-2 days for full dataset**

### 3. Storage Format for Output

**Option A: NumPy .npy** (recommended for moderate size)
- Simple, fast
- 10 GB for full 50k × 50k matrix
- Memory-maps efficiently

**Option B: HDF5** (for large scale)
```python
import h5py
with h5py.File('distances.h5', 'w') as f:
    f.create_dataset('distances', data=dist_matrix, compression='gzip')
```
- Compressed: ~3-5 GB
- Good for long-term storage
- Slower to access

**Option C: Sparse formats**
- Only if many zeros (unlikely for genetic distance)
- Check sparsity first

**Option D: Memory-mapped NumPy** (for huge matrices)
```python
dist_mmap = np.memmap('distances.dat', dtype='float32',
                      mode='w+', shape=(50000, 50000))
# Compute directly into mmap
```

### For 50k × 50k matrix:
- **Use .npy format**: Simple, 10 GB is manageable
- If storage is tight: Use HDF5 with compression

## Next Steps

1. **Short term** (if you need results soon):
   - Run CPU version on subsets
   - Process in batches if needed

2. **Medium term** (production setup):
   - Fix CUDA installation:
     ```bash
     conda create -n genodist python=3.10
     conda activate genodist
     conda install -c conda-forge cudatoolkit
     pip install torch numba
     ```
   - Benchmark GPU version on small data
   - Run full dataset on GPU

3. **Optimization ideas** (if GPU is still too slow):
   - Use multiple GPUs if available
   - Distributed computation across nodes
   - Approximate methods (random projections, sampling)

## Files Generated

- `generate_data.py` - Generate synthetic test data
- `distance_optimized.py` - Optimized CPU implementation (USE THIS)
- `distance_torch.py` - PyTorch GPU implementation
- `distance_numba.py` - Numba CUDA implementation
- `distance_matrix.py` - CuPy implementation + CPU baselines
- Test datasets: `test_small_*`, `test_medium_*`, `test_large_*`

## System Configuration

- CPU: 32 cores / 64 GB RAM
- GPU: NVIDIA GeForce RTX 4060 (8 GB) - CUDA 12.2
- Note: CUDA runtime libraries not properly linked in current environment
