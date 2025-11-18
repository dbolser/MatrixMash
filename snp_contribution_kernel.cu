/*
 * CUDA kernel for computing SNP-level contribution matrices in parallel.
 * Each grid plane along blockIdx.x corresponds to one SNP in the current batch.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// Tile size for the N x N distance plane. Tune per GPU.
#define TILE_SIZE 32

/*
 * Kernel to compute k SNP contribution matrices in parallel.
 *
 * Grid dimensions: (k, ceil(N / TILE_SIZE), ceil(N / TILE_SIZE))
 * Block dimensions: (1, TILE_SIZE, TILE_SIZE)
 *
 * Each block processes one TILE_SIZE x TILE_SIZE tile of a single SNP's N x N
 * contribution matrix. blockIdx.x selects the SNP within the batch.
 */
__global__ void compute_snp_contribution_kernel(
    const unsigned char* __restrict__ genotypes_batch,  // (k, N)
    const float* __restrict__ scores_batch,              // (k)
    float* __restrict__ contributions_out,               // (k, N, N)
    const int k,
    const int N
) {
    const int k_idx = blockIdx.x;  // Which SNP plane we're computing
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.z * blockDim.z + threadIdx.z;

    if (k_idx >= k) {
        // All threads in this block map to the same SNP. It is safe for the
        // entire block to exit early only when the whole block is out of range.
        return;
    }

    const bool in_bounds = (row < N) && (col < N);
    const bool compute_entry = in_bounds && (row <= col);

    // Shared memory for the TILE_SIZE entries along row/col for this SNP
    __shared__ unsigned char tile_rows[TILE_SIZE];
    __shared__ unsigned char tile_cols[TILE_SIZE];
    __shared__ float score_sq;

    if (threadIdx.y == 0 && threadIdx.z == 0) {
        const float score = scores_batch[k_idx];
        score_sq = score * score;
    }

    // Load the genotype row/column that this block works on.
    if (threadIdx.z == 0) {
        if (row < N) {
            tile_rows[threadIdx.y] = genotypes_batch[k_idx * N + row];
        }
    }
    if (threadIdx.y == 0) {
        if (col < N) {
            tile_cols[threadIdx.z] = genotypes_batch[k_idx * N + col];
        }
    }

    __syncthreads();

    if (compute_entry) {
        const unsigned char geno_i = tile_rows[threadIdx.y];
        const unsigned char geno_j = tile_cols[threadIdx.z];

        float contribution = 0.0f;
        if (geno_i != 9 && geno_j != 9) {
            const int diff = abs((int)((signed char)geno_i) - (int)((signed char)geno_j));
            if (diff == 1) {
                contribution = score_sq;
            } else if (diff == 2) {
                contribution = 4.0f * score_sq;
            }
        }

        const size_t slice_size = static_cast<size_t>(N) * N;
        const size_t base = static_cast<size_t>(k_idx) * slice_size;
        const size_t out_idx_ij = base + static_cast<size_t>(row) * N + col;
        const size_t out_idx_ji = base + static_cast<size_t>(col) * N + row;

        contributions_out[out_idx_ij] = contribution;
        contributions_out[out_idx_ji] = contribution;
    }
}

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

extern "C" {

/*
 * Host entry-point: copies host batch to device, launches kernel, copies
 * contributions back. Assumes the caller sizes the batch to fit device memory.
 */
void compute_snp_contributions_cuda(
    const unsigned char* h_genotypes_batch,  // (k, N)
    const float* h_scores_batch,             // (k)
    float* h_contributions_out,              // (k, N, N)
    int k,
    int N
) {
    unsigned char* d_genotypes_batch = nullptr;
    float* d_scores_batch = nullptr;
    float* d_contributions_out = nullptr;

    const size_t genotypes_size = static_cast<size_t>(k) * N * sizeof(unsigned char);
    const size_t scores_size = static_cast<size_t>(k) * sizeof(float);
    const size_t contributions_size = static_cast<size_t>(k) * N * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_genotypes_batch, genotypes_size));
    CUDA_CHECK(cudaMalloc(&d_scores_batch, scores_size));
    CUDA_CHECK(cudaMalloc(&d_contributions_out, contributions_size));

    CUDA_CHECK(cudaMemcpy(d_genotypes_batch, h_genotypes_batch, genotypes_size,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores_batch, h_scores_batch, scores_size,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_contributions_out, 0, contributions_size));

    const dim3 grid(k, (N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    const dim3 block(1, TILE_SIZE, TILE_SIZE);

    printf("Launching SNP contribution kernel: grid=(%d,%d,%d) block=(%d,%d,%d)\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z);
    printf("  Batch size k=%d, N=%d, output=%.2f MB\n", k, N,
           contributions_size / 1e6);

    compute_snp_contribution_kernel<<<grid, block>>>(
        d_genotypes_batch,
        d_scores_batch,
        d_contributions_out,
        k,
        N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_contributions_out, d_contributions_out, contributions_size,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_genotypes_batch));
    CUDA_CHECK(cudaFree(d_scores_batch));
    CUDA_CHECK(cudaFree(d_contributions_out));
}

}  // extern "C"
