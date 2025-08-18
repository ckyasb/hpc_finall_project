#include "winograd.cuh"
#include <thrust/device_vector.h>
#include <cstdio> 
#include <cstdlib>
#include <cublas_v2.h> // 引入 cuBLAS 库
#include <iostream>

// --- 变换矩阵 (常量内存) ---
__constant__ float G[4][3] = {
    {1.0f, 0.0f, 0.0f},
    {0.5f, 0.5f, 0.5f},
    {0.5f, -0.5f, 0.5f},
    {0.0f, 0.0f, 1.0f}
};

__constant__ float B_T[4][4] = {
    {1.0f, 0.0f, -1.0f, 0.0f}, 
    {0.0f, 1.0f, 1.0f, 0.0f}, 
    {0.0f, -1.0f, 1.0f, 0.0f}, 
    {0.0f, 1.0f, 0.0f, -1.0f}
};

__constant__ float B[4][4] = {
    {1.0f,  0.0f,  0.0f,  0.0f}, 
    {0.0f,  1.0f, -1.0f,  1.0f}, 
    {-1.0f, 1.0f,  1.0f,  0.0f}, 
    {0.0f,  0.0f,  0.0f, -1.0f}
};

__constant__ float A_T[2][4] = {
    {1.0f, 1.0f, 1.0f, 0.0f}, 
    {0.0f, 1.0f, -1.0f, -1.0f}
};

__constant__ float G_T[3][4] = {
    {1.0f, 0.5f, 0.5f, 0.0f},
    {0.0f, 0.5f, -0.5f, 0.0f},
    {0.0f, 0.5f, 0.5f, 1.0f}
};

__constant__ float A[4][2] = {
    {1.0f, 0.0f},
    {1.0f, 1.0f},
    {1.0f, -1.0f},
    {0.0f, -1.0f}
};


// =======================================================================
// =================== 方案A: 融合内核 (适用于小矩阵) ====================
// =======================================================================
__global__
void winograd_conv_fused_kernel(const float* __restrict__ image,
                                const float* __restrict__ filter,
                                float* __restrict__ output,
                                int N, int C, int H, int W, int K, int outH, int outW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tiles_w = (outW + 1) / 2;
    const int tiles_h = (outH + 1) / 2;
    const int tiles_per_image = tiles_h * tiles_w;
    const int num_tiles = N * K * tiles_per_image;
    
    if (idx >= num_tiles) return;

    // Decompose thread index to get (n, k, tile_y, tile_x)
    int p_local = idx % tiles_per_image;
    int k = (idx / tiles_per_image) % K;
    int n = idx / (K * tiles_per_image);
    int tile_y = p_local / tiles_w;
    int tile_x = p_local % tiles_w;

    float m[4][4] = {{0.0f}};

    // Loop over input channels
    for (int c = 0; c < C; ++c) {
        // --- Filter Transform ---
        const float* g = filter + (k * C + c) * 9;
        float u_kc[4][4];
        float temp_g[4][3];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 3; ++j) {
                temp_g[i][j] = G[i][0] * g[0 * 3 + j] + G[i][1] * g[1 * 3 + j] + G[i][2] * g[2 * 3 + j];
            }
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            u_kc[i][0] = temp_g[i][0];
            u_kc[i][1] = 0.5f * (temp_g[i][0] + temp_g[i][1] + temp_g[i][2]);
            u_kc[i][2] = 0.5f * (temp_g[i][0] - temp_g[i][1] + temp_g[i][2]);
            u_kc[i][3] = temp_g[i][2];
        }

        // --- Image Transform ---
        int h_start = tile_y * 2;
        int w_start = tile_x * 2;
        float d[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int h_in = h_start + i;
                int w_in = w_start + j;
                if (h_in < H && w_in < W) {
                    d[i][j] = image[(n * C + c) * H * W + h_in * W + w_in];
                } else {
                    d[i][j] = 0.0f;
                }
            }
        }
        float v_ncp[4][4];
        float temp_d[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                temp_d[i][j] = B_T[i][0] * d[0][j] + B_T[i][1] * d[1][j] + B_T[i][2] * d[2][j] + B_T[i][3] * d[3][j];
            }
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                v_ncp[i][j] = temp_d[i][0] * B[0][j] + temp_d[i][1] * B[1][j] + temp_d[i][2] * B[2][j] + temp_d[i][3] * B[3][j];
            }
        }

        // --- Element-wise product and accumulate ---
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                m[i][j] += u_kc[i][j] * v_ncp[i][j];
            }
        }
    }

    // --- Output Transform ---
    float temp_m[2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            temp_m[i][j] = A_T[i][0] * m[0][j] + A_T[i][1] * m[1][j] + A_T[i][2] * m[2][j] + A_T[i][3] * m[3][j];
        }
    }
    float Y[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        Y[i][0] = temp_m[i][0] + temp_m[i][1] + temp_m[i][2];
        Y[i][1] = temp_m[i][1] - temp_m[i][2] - temp_m[i][3];
    }

    // --- Write output ---
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int h = tile_y * 2 + i;
            int w = tile_x * 2 + j;
            if (h < outH && w < outW) {
                output[((n * K + k) * outH + h) * outW + w] = Y[i][j];
            }
        }
    }
}


// =======================================================================
// ============ 方案B: 多内核+Tiling+cuBLAS (适用于大矩阵) ==============
// =======================================================================

// --- Kernel 1: 滤波器变换 (Grid-Stride Loop) ---
__global__ void transform_filter_kernel_merged(const float* __restrict__ filter, float* __restrict__ U_padded, int C, int K, int padded_C, int padded_K) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < K * C; idx += gridDim.x * blockDim.x) {
        int k = idx / C;
        int c = idx % C;
        const float* g = filter + (k * C + c) * 9;
        float temp_g[4][3];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 3; ++j) {
                temp_g[i][j] = G[i][0] * g[j] + G[i][1] * g[3 + j] + G[i][2] * g[6 + j];
            }
        }
        float u_kc[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                u_kc[i][j] = temp_g[i][0] * G_T[0][j] + temp_g[i][1] * G_T[1][j] + temp_g[i][2] * G_T[2][j];
            }
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int gemm_idx = i * 4 + j;
                U_padded[gemm_idx * padded_K * padded_C + k * padded_C + c] = u_kc[i][j];
            }
        }
    }
}

// --- Kernel 2: 输入变换 (高级Tiling策略) ---
constexpr int TILE_W = 32;
constexpr int TILE_H = 32;
constexpr int PATCH_W = TILE_W * 2 + 2;
constexpr int PATCH_H = TILE_H * 2 + 2;

__global__ void transform_input_kernel_tiled(const float* __restrict__ image, float* __restrict__ V_padded, int N, int C, int H, int W, int P, int outH, int outW, int padded_C, int padded_P) {
    __shared__ float patch[PATCH_H][PATCH_W];
    const int tile_x_base = blockIdx.x * TILE_W;
    const int tile_y_base = blockIdx.y * TILE_H;
    const int nc = blockIdx.z;
    const int n = nc / C;
    const int c = nc % C;
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int h_start_base = tile_y_base * 2;
    const int w_start_base = tile_x_base * 2;
    const int block_size = TILE_W * TILE_H;
    const int thread_id_in_block = thy * TILE_W + thx;
    for (int i = thread_id_in_block; i < PATCH_H * PATCH_W; i += block_size) {
        const int h = i / PATCH_W;
        const int w = i % PATCH_W;
        const int h_in = h_start_base + h;
        const int w_in = w_start_base + w;
        if (h_in < H && w_in < W) {
            patch[h][w] = image[(n * C + c) * H * W + h_in * W + w_in];
        } else {
            patch[h][w] = 0.0f;
        }
    }
    __syncthreads();
    const int tile_x = tile_x_base + thx;
    const int tile_y = tile_y_base + thy;
    const int tiles_w = (outW + 1) / 2;
    if (tile_x < tiles_w && tile_y < (outH + 1) / 2) {
        float d[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                d[i][j] = patch[thy * 2 + i][thx * 2 + j];
            }
        }
        float temp_d[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                temp_d[i][j] = B_T[i][0] * d[0][j] + B_T[i][1] * d[1][j] + B_T[i][2] * d[2][j] + B_T[i][3] * d[3][j];
            }
        }
        float v_ncp[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                v_ncp[i][j] = temp_d[i][0] * B[0][j] + temp_d[i][1] * B[1][j] + temp_d[i][2] * B[2][j] + temp_d[i][3] * B[3][j];
            }
        }
        const int tiles_per_image = ((outH + 1) / 2) * tiles_w;
        const int p_idx = n * tiles_per_image + tile_y * tiles_w + tile_x;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int gemm_idx = i * 4 + j;
                V_padded[gemm_idx * padded_C * padded_P + c * padded_P + p_idx] = v_ncp[i][j];
            }
        }
    }
}

// --- Kernel 4: 输出变换 (Grid-Stride Loop) ---
__global__ void transform_output_kernel(const float* __restrict__ M_padded, float* __restrict__ output, int N, int K, int P, int outH, int outW, int padded_K, int padded_P) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < K * P; idx += gridDim.x * blockDim.x) {
        int k = idx / P;
        int p_idx = idx % P;
        float m[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                m[i][j] = M_padded[(i * 4 + j) * padded_K * padded_P + k * padded_P + p_idx];
            }
        }
        float temp_m[2][4];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                temp_m[i][j] = A_T[i][0] * m[0][j] + A_T[i][1] * m[1][j] + A_T[i][2] * m[2][j] + A_T[i][3] * m[3][j];
            }
        }
        float Y_tile[2][2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                Y_tile[i][j] = temp_m[i][0] * A[0][j] + temp_m[i][1] * A[1][j] + temp_m[i][2] * A[2][j] + temp_m[i][3] * A[3][j];
            }
        }
        const int tiles_w = (outW + 1) / 2;
        const int tiles_per_image = ((outH + 1) / 2) * tiles_w;
        const int n = p_idx / tiles_per_image;
        const int p_local = p_idx % tiles_per_image;
        const int tile_y = p_local / tiles_w;
        const int tile_x = p_local % tiles_w;
        int h_start = tile_y * 2;
        int w_start = tile_x * 2;
        #pragma unroll
        for(int i = 0; i < 2; ++i) {
            #pragma unroll
            for(int j = 0; j < 2; ++j) {
                int h = h_start + i;
                int w = w_start + j;
                if (h < outH && w < outW) {
                    int out_idx = n * K * outH * outW + k * outH * outW + h * outW + w;
                    output[out_idx] = Y_tile[i][j];
                }
            }
        }
    }
}

// --- Helper function to run the Winograd pipeline on a single GPU ---
void winograd_pipeline_single_gpu(const float* d_image, const float* d_filter, float* d_out,
                                  int H, int W, int C, int K, int N, cudaStream_t stream) {
    const int outH = H - 2;
    const int outW = W - 2;

    // --- 启发式策略 ---
    if (C < 16) {
        const int threads_per_block = 256;
        const int tiles_w = (outW + 1) / 2;
        const int tiles_h = (outH + 1) / 2;
        const int num_tiles = N * K * tiles_h * tiles_w;
        int grid_size = (num_tiles + threads_per_block - 1) / threads_per_block;
        winograd_conv_fused_kernel<<<grid_size, threads_per_block, 0, stream>>>(d_image, d_filter, d_out, N, C, H, W, K, outH, outW);
        return;
    }

    // --- Winograd + cuBLAS 流程 (适用于大矩阵) ---
    const int P = N * ((outH + 1) / 2) * ((outW + 1) / 2);
    if (P == 0) {
        cudaMemsetAsync(d_out, 0, N * K * outH * outW * sizeof(float), stream);
        return;
    }
    int padded_C = (C + 15) & ~15;
    int padded_K = (K + 15) & ~15;
    int padded_P = (P + 15) & ~15;

    float *U_padded, *V_padded, *M_padded;
    cudaMalloc(&U_padded, 16 * padded_K * padded_C * sizeof(float));
    cudaMalloc(&V_padded, 16 * padded_C * padded_P * sizeof(float));
    cudaMalloc(&M_padded, 16 * padded_K * padded_P * sizeof(float));
    cudaMemsetAsync(U_padded, 0, 16 * padded_K * padded_C * sizeof(float), stream);
    cudaMemsetAsync(V_padded, 0, 16 * padded_C * padded_P * sizeof(float), stream);
    cudaMemsetAsync(M_padded, 0, 16 * padded_K * padded_P * sizeof(float), stream);

    const int threads_per_block = 256;
    const int total_filters = K * C;
    int grid_size_filter = (total_filters + threads_per_block - 1) / threads_per_block;
    transform_filter_kernel_merged<<<grid_size_filter, threads_per_block, 0, stream>>>(d_filter, U_padded, C, K, padded_C, padded_K);

    dim3 block_dim_input(TILE_W, TILE_H);
    const int num_tiles_w = (outW + 1) / 2;
    const int num_tiles_h = (outH + 1) / 2;
    dim3 grid_dim_input((num_tiles_w + TILE_W - 1) / TILE_W, (num_tiles_h + TILE_H - 1) / TILE_H, N * C);
    transform_input_kernel_tiled<<<grid_dim_input, block_dim_input, 0, stream>>>(d_image, V_padded, N, C, H, W, P, outH, outW, padded_C, padded_P);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int batch_count = 16;
    const float** h_U_array = new const float*[batch_count];
    const float** h_V_array = new const float*[batch_count];
    float** h_M_array = new float*[batch_count];
    for(int i = 0; i < batch_count; ++i) {
        h_U_array[i] = U_padded + i * padded_K * padded_C;
        h_V_array[i] = V_padded + i * padded_C * padded_P;
        h_M_array[i] = M_padded + i * padded_K * padded_P;
    }
    const float **d_U_array, **d_V_array;
    float **d_M_array;
    cudaMalloc(&d_U_array, batch_count * sizeof(float*));
    cudaMalloc(&d_V_array, batch_count * sizeof(float*));
    cudaMalloc(&d_M_array, batch_count * sizeof(float*));
    cudaMemcpyAsync(d_U_array, h_U_array, batch_count * sizeof(float*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_V_array, h_V_array, batch_count * sizeof(float*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_M_array, h_M_array, batch_count * sizeof(float*), cudaMemcpyHostToDevice, stream);
    
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       padded_P, padded_K, padded_C,
                       &alpha,
                       d_V_array, padded_P,
                       d_U_array, padded_C,
                       &beta,
                       d_M_array, padded_P,
                       batch_count);
    delete[] h_U_array;
    delete[] h_V_array;
    delete[] h_M_array;
    cudaFree(d_U_array);
    cudaFree(d_V_array);
    cudaFree(d_M_array);
    cublasDestroy(handle);

    const int total_output_tiles = K * P;
    int grid_size_output = (total_output_tiles + threads_per_block - 1) / threads_per_block;
    transform_output_kernel<<<grid_size_output, threads_per_block, 0, stream>>>(M_padded, d_out, N, K, P, outH, outW, padded_K, padded_P);
    
    cudaFree(U_padded);
    cudaFree(V_padded);
    cudaFree(M_padded);
}


// --- 主机函数: Winograd 卷积调度器 (双卡并行) ---
void winograd_conv(thrust::device_vector<float>& image,
                   thrust::device_vector<float>& filter,
                   thrust::device_vector<float>& out,
                   thrust::device_vector<float>& U, // Unused
                   thrust::device_vector<float>& V, // Unused
                   thrust::device_vector<float>& M, // Unused
                   int H, int W, int C, int K, int N) {
    
    int device_count;
    cudaGetDeviceCount(&device_count);

    if (device_count < 2 || N < 2) { // 如果少于2个GPU或批次太小，则回退到单卡
        cudaSetDevice(0);
        winograd_pipeline_single_gpu(image.data().get(), filter.data().get(), out.data().get(), H, W, C, K, N, 0);
        cudaDeviceSynchronize();
        return;
    }

    // 启用GPU 0和GPU 1之间的P2P内存访问
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);

    // 创建流
    cudaStream_t stream0, stream1;
    cudaSetDevice(0);
    cudaStreamCreate(&stream0);
    cudaSetDevice(1);
    cudaStreamCreate(&stream1);

    // 划分批次
    int N0 = N / 2;
    int N1 = N - N0;
    size_t image_size_bytes = (size_t)C * H * W * sizeof(float);
    size_t filter_size_bytes = (size_t)K * C * 9 * sizeof(float);
    const int outH = H - 2;
    const int outW = W - 2;
    size_t out_size_bytes = (size_t)K * outH * outW * sizeof(float);

    // --- GPU 1 任务 ---
    cudaSetDevice(1);
    float *d_image1, *d_filter1, *d_out1;
    cudaMalloc(&d_image1, N1 * image_size_bytes);
    cudaMalloc(&d_filter1, filter_size_bytes);
    cudaMalloc(&d_out1, N1 * out_size_bytes);
    
    // 从GPU 0异步复制数据到GPU 1
    cudaMemcpyPeerAsync(d_filter1, 1, filter.data().get(), 0, filter_size_bytes, stream1);
    cudaMemcpyPeerAsync(d_image1, 1, image.data().get() + N0 * C * H * W, 0, N1 * image_size_bytes, stream1);

    // 在GPU 1上启动计算流水线
    winograd_pipeline_single_gpu(d_image1, d_filter1, d_out1, H, W, C, K, N1, stream1);

    // 将GPU 1的结果异步复制回GPU 0上的主输出Buffer
    cudaMemcpyPeerAsync(out.data().get() + N0 * K * outH * outW, 0, d_out1, 1, N1 * out_size_bytes, stream1);

    // --- GPU 0 任务 ---
    cudaSetDevice(0);
    // 在GPU 0上启动计算流水线 (数据已在主输入Buffer中)
    winograd_pipeline_single_gpu(image.data().get(), filter.data().get(), out.data().get(), H, W, C, K, N0, stream0);

    // --- 同步与清理 ---
    cudaSetDevice(0);
    cudaStreamSynchronize(stream0);
    cudaSetDevice(1);
    cudaStreamSynchronize(stream1);

    cudaFree(d_image1);
    cudaFree(d_filter1);
    cudaFree(d_out1);

    cudaSetDevice(0);
    cudaStreamDestroy(stream0);
    cudaSetDevice(1);
    cudaStreamDestroy(stream1);
}

