/*
 * CUDA kernel for computing genotype distance matrix
 * Optimized for memory-bound workload with shared memory and kernel fusion
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Shared memory tile size - tune based on GPU
#define TILE_SIZE 32
#define GENOTYPE_CHUNK 512  // Process this many genotypes at once in shared mem (32*512*2 + 512*4 = 34KB)

/*
 * Main distance kernel - processes one block of the distance matrix
 * Each block computes a TILE_SIZE × TILE_SIZE submatrix of distances
 *
 * Grid dimensions: (N/TILE_SIZE, N/TILE_SIZE)
 * Block dimensions: (TILE_SIZE, TILE_SIZE)
 *
 * Each thread computes one distance[i,j] by accumulating over all M genotypes
 */
__global__ void compute_distance_kernel(
    const unsigned char* __restrict__ genotypes,  // M × N (row-major)
    const float* __restrict__ scores,              // M
    float* __restrict__ distances,                 // N × N (output)
    const int M,
    const int N
) {
    // Global indices for this thread's distance matrix element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Only compute upper triangle (including diagonal)
    if (row >= N || col >= N || row > col) return;

    // Shared memory for genotype tiles
    __shared__ unsigned char tile_i[TILE_SIZE][GENOTYPE_CHUNK];
    __shared__ unsigned char tile_j[TILE_SIZE][GENOTYPE_CHUNK];
    __shared__ float tile_scores[GENOTYPE_CHUNK];

    float distance = 0.0f;

    // Process genotypes in chunks that fit in shared memory
    for (int m_start = 0; m_start < M; m_start += GENOTYPE_CHUNK) {
        int m_end = min(m_start + GENOTYPE_CHUNK, M);
        int chunk_size = m_end - m_start;

        // Load scores (first row of threads cooperate)
        if (threadIdx.y == 0) {
            for (int m = 0; m < chunk_size; m++) {
                if (threadIdx.x == 0) {
                    tile_scores[m] = scores[m_start + m];
                }
            }
        }

        // Load genotypes into shared memory
        // Strategy: Each thread loads data for its own position
        int global_row = blockIdx.y * blockDim.y + threadIdx.y;
        int global_col = blockIdx.x * blockDim.x + threadIdx.x;

        // Each thread loads its row's genotypes
        if (global_row < N) {
            for (int m = 0; m < chunk_size; m++) {
                int global_m = m_start + m;
                tile_i[threadIdx.y][m] = genotypes[global_m * N + global_row];
            }
        }

        // Each thread loads its column's genotypes
        if (global_col < N) {
            for (int m = 0; m < chunk_size; m++) {
                int global_m = m_start + m;
                tile_j[threadIdx.x][m] = genotypes[global_m * N + global_col];
            }
        }

        __syncthreads();

        // Compute distance contribution from this chunk
        #pragma unroll 8
        for (int m = 0; m < chunk_size; m++) {
            unsigned char geno_i = tile_i[threadIdx.y][m];
            unsigned char geno_j = tile_j[threadIdx.x][m];

            // Skip unknown genotypes (9)
            if (geno_i == 9 || geno_j == 9) continue;

            // Compute absolute difference (explicit signed cast to avoid unsigned underflow)
            int diff = abs((int)((signed char)geno_i) - (int)((signed char)geno_j));

            // Distance contribution: 0 for same, s^2 for het (diff=1), 4*s^2 for hom (diff=2)
            if (diff == 1) {
                float s = tile_scores[m];
                distance += s * s;
            } else if (diff == 2) {
                float s = tile_scores[m];
                distance += 4.0f * s * s;
            }
        }

        __syncthreads();
    }

    // Write result (symmetric)
    distances[row * N + col] = distance;
    if (row != col) {
        distances[col * N + row] = distance;
    }
}

/*
 * Optimized kernel for diagonal blocks where row and col tiles are the same
 * Can reuse loaded data and only compute upper triangle within block
 */
__global__ void compute_distance_kernel_diagonal(
    const unsigned char* __restrict__ genotypes,
    const float* __restrict__ scores,
    float* __restrict__ distances,
    const int M,
    const int N,
    const int block_offset  // Starting index for this diagonal block
) {
    int row = block_offset + threadIdx.y;
    int col = block_offset + threadIdx.x;

    if (row >= N || col >= N || row > col) return;

    __shared__ unsigned char tile_geno[TILE_SIZE][GENOTYPE_CHUNK];
    __shared__ float tile_scores[GENOTYPE_CHUNK];

    float distance = 0.0f;

    for (int m_start = 0; m_start < M; m_start += GENOTYPE_CHUNK) {
        int m_end = min(m_start + GENOTYPE_CHUNK, M);
        int chunk_size = m_end - m_start;

        // Load scores and genotypes
        if (threadIdx.y == 0) {
            for (int m = 0; m < chunk_size; m++) {
                tile_scores[m] = scores[m_start + m];
            }
        }

        // Each thread loads its row's genotypes
        if (row < N) {
            for (int m = 0; m < chunk_size; m++) {
                int global_m = m_start + m;
                tile_geno[threadIdx.y][m] = genotypes[global_m * N + row];
            }
        }

        __syncthreads();

        #pragma unroll 8
        for (int m = 0; m < chunk_size; m++) {
            unsigned char geno_i = tile_geno[threadIdx.y][m];
            unsigned char geno_j = tile_geno[threadIdx.x][m];

            if (geno_i == 9 || geno_j == 9) continue;

            int diff = abs((int)((signed char)geno_i) - (int)((signed char)geno_j));

            if (diff == 1) {
                float s = tile_scores[m];
                distance += s * s;
            } else if (diff == 2) {
                float s = tile_scores[m];
                distance += 4.0f * s * s;
            }
        }

        __syncthreads();
    }

    distances[row * N + col] = distance;
    if (row != col) {
        distances[col * N + row] = distance;
    }
}

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

extern "C" {

/*
 * Main entry point callable from Python
 *
 * genotypes: M × N uint8 array (row-major, M genotypes × N individuals)
 * scores: M float32 array
 * distances: N × N float32 array (output, pre-allocated)
 * M: number of genotypes
 * N: number of individuals
 */
void compute_distance_cuda(
    const unsigned char* genotypes,
    const float* scores,
    float* distances,
    int M,
    int N
) {
    // Allocate device memory
    unsigned char *d_genotypes;
    float *d_scores;
    float *d_distances;

    size_t genotypes_size = (size_t)M * N * sizeof(unsigned char);
    size_t scores_size = M * sizeof(float);
    size_t distances_size = (size_t)N * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_genotypes, genotypes_size));
    CUDA_CHECK(cudaMalloc(&d_scores, scores_size));
    CUDA_CHECK(cudaMalloc(&d_distances, distances_size));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_genotypes, genotypes, genotypes_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, scores, scores_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_distances, 0, distances_size));

    // Launch configuration
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    printf("Launching kernel with grid (%d, %d), block (%d, %d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    printf("Processing %d genotypes × %d individuals\n", M, N);
    printf("Memory: genotypes=%.1f MB, scores=%.1f MB, distances=%.1f MB\n",
           genotypes_size/1e6, scores_size/1e6, distances_size/1e6);

    // Launch kernel - process all blocks with general kernel
    // (Could optimize by using diagonal kernel for diagonal blocks)
    compute_distance_kernel<<<gridDim, blockDim>>>(
        d_genotypes, d_scores, d_distances, M, N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(distances, d_distances, distances_size, cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_genotypes));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_distances));
}

} // extern "C"
