# Distributed Computation Strategy

## Overview

The distance matrix computation is **embarrassingly parallel** - perfect for distributed cloud computation. Each machine computes an independent chunk of the output matrix with no inter-machine communication needed during processing.

## Why Distribute?

**The bottleneck is memory bandwidth, not CPU cores.**

- Single 100-core machine: Still limited by memory bandwidth (~9 days)
- 100 machines with 16 cores each: **100× the aggregate memory bandwidth** (~2-3 hours)
- Cost is similar, but execution is 30-50× faster

## Problem Decomposition

### The Task
- Input: M × N genotype matrix (1,000,000 × 50,000)
- Output: N × N distance matrix (50,000 × 50,000)
- Complexity: 1.25 × 10¹⁵ pair-genotype comparisons

### Distribution Strategy

**Split the distance matrix by rows** - each machine computes a subset of individual pairs:

```
Machine k of K total machines computes:
- Rows: [i_start, i_end) where i_start = (N * k) // K
- Full row j from 0 to N for each i in its range
- Output: chunk of size (i_end - i_start) × N
```

**Why this works:**
- ✅ Each pair (i,j) computed by exactly one machine
- ✅ No communication during computation
- ✅ Load is balanced (all machines do similar work)
- ✅ Each machine needs same input (full genotype matrix)
- ✅ Results are independent - just concatenate at end

### Data Requirements Per Machine

**Input (read-only, same for all machines):**
- Genotype matrix: 1M × 50k × 1 byte = 50 GB
- Scores: 1M × 4 bytes = 4 MB
- Total: ~50 GB (can use shared storage or replicate)

**Output (per machine with K machines):**
- Distance chunk: (50k/K) × 50k × 4 bytes
- Example with K=100: 500 × 50k × 4 = 100 MB per machine
- Total across all machines: 10 GB

## Cloud Deployment Options

### Option A: CPU Instances (Simple, Reliable)

**Instance Type:** `c5.4xlarge` (16 vCPU, 32 GB RAM)

**Configuration:**
```
Number of instances: 100
Rows per instance: 500 (out of 50,000)
Runtime: ~2.5-3 hours
```

**Scaling estimate:**
- Single 32-core machine: ~9 days = 216 hours
- 100× 16-core instances: 216 / 100 × (16/32 speedup) × (efficiency factor ~1.5) ≈ 2.5-3 hours
  - (Efficiency factor accounts for better per-core performance on smaller machines)

**Cost:**
- Hourly rate: $0.68/hour (us-east-1)
- Total: 100 instances × $0.68/hr × 3 hrs = **~$204**
- Add ~$20 for data transfer and storage = **~$224 total**

**Pros:**
- Simple, proven approach
- No GPU setup/debugging
- Plenty of RAM for data
- Spot instances available (~70% cheaper if time-flexible)

**Cons:**
- Still relatively slow (3 hours)
- Higher total cost

---

### Option B: GPU Instances (Faster, Cheaper)

**Instance Type:** `g4dn.xlarge` (4 vCPU, 16 GB RAM, NVIDIA T4 GPU)

**Configuration:**
```
Number of instances: 10-20
Rows per instance: 5,000-2,500 (out of 50,000)
Runtime: ~0.5-1 hour
```

**Scaling estimate:**
- GPU is 10-50× faster than CPU for this workload
- Conservative estimate: 20× speedup
- 10 instances × 1 hour (with 20× GPU speedup) ≈ 216 hours / (10 × 20) ≈ 1 hour

**Cost (10 instances):**
- Hourly rate: $0.526/hour (us-east-1)
- Total: 10 instances × $0.526/hr × 1 hr = **~$5.26**
- Add ~$10 for data transfer and storage = **~$15 total**

**Cost (20 instances for redundancy/speed):**
- Total: 20 instances × $0.526/hr × 0.5 hr = **~$5.26**
- Add ~$10 for data transfer = **~$15 total**

**Pros:**
- **15× cheaper** than CPU option
- Much faster (30-60 minutes vs 3 hours)
- Proven GPU implementations ready to use
- Modern NVIDIA T4 GPU with good CUDA support

**Cons:**
- Requires GPU code setup (already written, just needs testing)
- Less RAM (16 GB) - need to stream data or use shared storage
- Slightly more complex orchestration

---

### Option C: Hybrid Approach

Use CPU instances but exploit spot pricing:

**Configuration:**
```
Instance type: c5.4xlarge (spot)
Number of instances: 100-150
Spot price: ~$0.20/hr (70% discount)
Runtime: 2-3 hours
Total cost: 100 × $0.20 × 3 = ~$60
```

**Pros:**
- Very cost-effective
- Simple CPU code (no GPU debugging)

**Cons:**
- Instances can be interrupted (need checkpointing)
- Availability not guaranteed
- Still slower than GPU

## Recommendation

**🏆 Best Option: GPU Instances (Option B)**

Deploy **10-20 × g4dn.xlarge** instances with GPU implementation:

- **Cost**: ~$15 total
- **Time**: 30-60 minutes
- **Risk**: Low (GPU code already written, just needs CUDA setup)

**Why GPU wins:**
- 15× cheaper than CPU approach
- 3-6× faster execution
- Memory bandwidth is the key bottleneck - GPU excels here
- T4 GPU has 320 GB/s memory bandwidth vs ~100 GB/s for DDR4
- CUDA toolkit bundled with cloud GPU images (no setup issues)

**Fallback:** If GPU debugging takes >1 day, switch to CPU instances (Option A)

## Implementation Checklist

### 1. Orchestration Setup
- [ ] Choose cloud provider (AWS, GCP, Azure)
- [ ] Set up data storage (S3/GCS/Blob)
  - Upload genotype matrix (50 GB)
  - Upload scores (4 MB)
- [ ] Create job distribution system
  - Assign row ranges to workers
  - Track completion status

### 2. Worker Script
- [ ] Load genotypes and scores from shared storage
- [ ] Compute assigned row range
- [ ] Save output chunk to storage
- [ ] Report completion

### 3. Result Aggregation
- [ ] Collect all output chunks
- [ ] Concatenate into full N×N matrix
- [ ] Validate (check symmetry, expected range)
- [ ] Save final result

### 4. Fault Tolerance
- [ ] Implement retries for failed workers
- [ ] Add progress monitoring
- [ ] Optional: Checkpoint intermediate results

## Cost-Optimization Tips

1. **Use spot/preemptible instances**
   - 60-80% cheaper
   - Acceptable for short jobs with retries

2. **Use cheaper regions**
   - us-east-1 is usually cheapest
   - Check regional pricing

3. **Pre-warm instances**
   - Create custom AMI/image with dependencies
   - Faster startup, less waste

4. **Right-size instances**
   - Don't over-provision memory
   - Match to actual working set

5. **Use object storage efficiently**
   - Memory-map large files
   - Stream data instead of copying

## Alternative: Pre-existing Frameworks

If you don't want to roll your own:

**Dask / Ray**
- Distributed Python frameworks
- Handle orchestration, fault tolerance
- Can use your existing Python code
- Work on any cloud provider

**AWS Batch / Google Cloud Batch**
- Managed job scheduling
- Auto-scaling workers
- Built-in retry logic
- ~$0 orchestration overhead

**Kubernetes + Job Controller**
- More complex but very flexible
- Good if you already have K8s expertise

## Example Row Distribution (100 machines)

```
Machine 0:  rows 0-499
Machine 1:  rows 500-999
Machine 2:  rows 1000-1499
...
Machine 99: rows 49500-49999
```

Each machine:
1. Loads full genotype matrix (50 GB)
2. Loads scores (4 MB)
3. Computes distance for its 500 rows × 50k columns
4. Saves 500 × 50k × 4 bytes = 100 MB output
5. Uploads to cloud storage
6. Terminates

Final step: Concatenate all 100 chunks → 50k × 50k matrix

## Summary

| Approach | Instances | Runtime | Cost | Complexity |
|----------|-----------|---------|------|------------|
| **GPU (recommended)** | 10-20 × g4dn.xlarge | 30-60 min | ~$15 | Medium |
| CPU On-Demand | 100 × c5.4xlarge | 2-3 hours | ~$224 | Low |
| CPU Spot | 100 × c5.4xlarge | 2-3 hours | ~$60 | Medium |
| Single Machine | 1 × c5.24xlarge | 9 days | ~$1500 | Low |

**GPU option is 15× cheaper and 3-6× faster than CPU distributed approach.**

The embarrassingly parallel nature of this problem makes it perfect for cloud distribution - you get linear scaling in aggregate memory bandwidth, which is exactly what you need for this memory-bound workload.
