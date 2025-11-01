# Scaling Analysis - CPU and GPU

## Executive Summary

**CPU: Sublinear scaling, memory bandwidth limited**
- 32 cores → **9.5x speedup** (30% efficiency)
- 8 cores → **3.9x speedup** (49% efficiency)
- Full dataset: **9 days** on 32-core machine

**GPU: Better scaling, 3-6× faster than 32-core CPU**
- RTX 4060 Laptop (8GB): **3.5 days** for full dataset
- Linear scaling with M (genotypes)
- 64% efficiency with N (individuals)
- Throughput: 3,900-4,700 M comparisons/sec (vs 700-1,600 on CPU)

---

## CPU Core Scaling

### Summary

**Scaling is SUBLINEAR and MEMORY BANDWIDTH LIMITED**

- 32 cores → **9.5x speedup** (30% efficiency)
- 8 cores → **3.9x speedup** (49% efficiency)
- Memory bandwidth saturates around 8-16 cores

## Detailed Results

Test on 50k genotypes × 1k individuals:

| Threads | Time (s) | Speedup | Efficiency |
|---------|----------|---------|------------|
| 1       | 145.5    | 1.00x   | 100%       |
| 2       | 106.6    | 1.36x   | 68%        |
| 4       | 61.0     | 2.38x   | 60%        |
| 8       | 37.3     | 3.91x   | 49%        |
| 16      | 23.5     | 6.19x   | 39%        |
| 32      | 15.3     | 9.48x   | 30%        |

## Why Is Scaling Sublinear?

**Your algorithm is memory bandwidth limited:**

1. **High memory traffic**: For each pair of individuals (i,j), you read M genotype values
   - With M=1M, that's 1 million memory reads per pair
   - Total memory reads: N×N×M = 50k × 50k × 1M = 2.5 × 10¹⁵ bytes

2. **Low computation per byte**: Only simple operations (diff, comparison, add)
   - Arithmetic intensity is very low (~1-2 FLOPs per byte read)

3. **Memory bandwidth saturation**:
   - Adding threads beyond 8-16 doesn't help because they all compete for same memory bus
   - DDR4/DDR5 bandwidth is shared across all cores

4. **Not cache-friendly**:
   - Genotype matrix is huge (50 GB for full dataset)
   - Random access pattern across genotypes
   - L3 cache is only ~32-64 MB

## Implications for Full Dataset

### With 32 cores (your system):
- Expected time: **8.9 days** (using scaling test extrapolation)
- This matches our earlier 9-20 day estimate ✓

### Throwing more CPU cores won't help much:
- 64 cores would give maybe 12-15x speedup (not 20x)
- Diminishing returns due to memory bandwidth

---

## GPU Scaling (Actual Benchmarks)

### Test Results - RTX 4060 Laptop GPU (8GB)

**Hardware:**
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Memory: 8GB GDDR6, 272 GB/s bandwidth
- Implementation: PyTorch CUDA with chunked processing

**N Scaling (M=10k fixed):**

| N     | Time (s) | Throughput (pairs/sec) | Comparisons/sec (M) |
|-------|----------|------------------------|---------------------|
| 1,000 | 1.66     | 302,059                | 3,021               |
| 2,000 | 4.89     | 409,050                | 4,091               |
| 3,000 | 10.20    | 441,312                | 4,413               |
| 4,000 | 17.52    | 456,852                | 4,569               |
| 5,000 | 26.64    | 469,381                | 4,694               |

- Expected O(N²) ratio (1k→5k): 25.0×
- Actual ratio: 16.1×
- **Efficiency: 64%** ✅

**M Scaling (N=2k fixed):**

| M      | Time (s) | Comparisons/sec (M) |
|--------|----------|---------------------|
| 10,000 | 4.71     | 4,245               |
| 25,000 | 11.74    | 4,263               |
| 50,000 | 25.59    | 3,910               |

- Expected O(M) ratio (10k→50k): 5.0×
- Actual ratio: 5.4×
- **Efficiency: ~92%** (near-linear!) ✅

### Key Findings

**GPU is 3-6× faster than 32-core CPU:**
- GPU throughput: 3,900-4,700 M comparisons/sec
- CPU throughput: 700-1,600 M comparisons/sec
- **Speedup: 3-6×** (consistent across problem sizes)

**GPU scales linearly with M (genotypes):**
- Maintains ~4,000 M comparisons/sec as M increases
- CPU degrades due to memory bandwidth saturation
- This is the key advantage

**GPU efficiency improves with larger N:**
- Throughput increases from 302k → 469k pairs/sec (1k→5k individuals)
- Better GPU utilization at scale
- CPU efficiency decreases with more cores

### Extrapolation to Full Dataset

**Method 1: Scale from M=50k, N=2k test**
```
Base: 25.59 seconds for 50k × 2k
Scale: (50k/2k)² × (1M/50k) = 625 × 20 = 12,500×
Estimated: 25.59 × 12,500 = 319,875 sec = 88.9 hours = 3.7 days
```

**Method 2: Use sustained throughput**
```
Throughput: 3,910 M comparisons/sec
Total: 1.25 × 10¹⁵ comparisons
Estimated: 319,693 sec = 88.8 hours = 3.7 days
```

**GPU Estimate: 3.5-3.7 days** (vs 9 days on 32-core CPU)

### Why GPU Speedup is "Only" 3-6×

The 2.7× memory bandwidth advantage (272 vs 100 GB/s) translates to 3-6× real-world speedup because:

1. **Better latency hiding**: GPU can overlap 1000s of memory requests
2. **Compute is free**: Arithmetic happens while waiting for memory
3. **Linear M scaling**: GPU doesn't saturate bandwidth like CPU does

But it's not 10-50× because:
1. **Still memory-bound**: Can't exceed bandwidth limits
2. **Small GPU (8GB)**: Limited chunk sizes, more overhead
3. **Memory transfer overhead**: CPU↔GPU data movement

**With larger cloud GPUs (T4 16GB, A100 40GB)**, speedup could be 10-20× due to:
- Larger chunks → better utilization
- Higher bandwidth (A100: 1555 GB/s → potential 15× CPU)
- Less transfer overhead

## Optimization Strategies

If you need faster CPU performance:

### 1. **Block/Tile the computation** (cache-friendly)
```
For each block of genotypes (e.g., 10k at a time):
    Load into cache
    Compute partial distances for all pairs
    Accumulate
```
Might improve scaling to 15-20x on 32 cores.

### 2. **Transpose the matrix**
Store individuals in rows instead of columns for better cache locality.

### 3. **Use SIMD intrinsics**
Manually vectorize the inner loop (8-16 genotypes at once).

### 4. **Distributed computation**
- Split pairs across multiple machines
- Each computes its portion of the distance matrix
- Combine at the end

### 5. **Approximation methods**
- Random projection / sketching
- Sample subset of genotypes
- Dimension reduction

## Bottom Line

**CPU scaling is limited by memory bandwidth, not computation.**

- ✅ Using all 32 cores: ~9 days
- ❌ Using more cores won't help much
- ✅ GPU will be 10-50x faster: 0.5-2 days
- ✅ Best ROI: Fix CUDA setup and use GPU

The algorithm is fundamentally memory-bound, so focus on:
1. Higher memory bandwidth (GPU)
2. Better cache utilization (blocking/tiling)
3. Reducing data movement (approximation)
