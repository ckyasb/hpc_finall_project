#include "winograd.cuh"
#include <thrust/device_vector.h>
#include <cstdio> 
#include <cstdlib>
#include <cublas_v2.h> // 引入 cuBLAS 库

// --- 变换矩阵 (常量内存) ---
__constant__ float G[4][3] = {
    {1.0f, 0.0f, 0.0f},
    {0.5f, 0.5f, 0.5f},
    {0.5f, -0.5f, 0.5f},
    {0.0f, 0.0f, 1.0f}
};

__constant__ float G_T[3][4] = {
    {1.0f, 0.5f, 0.5f, 0.0f},
    {0.0f, 0.5f, -0.5f, 0.0f},
    {0.0f, 0.5f, 0.5f, 1.0f}
};

__constant__ float B_T[4][4] = {
    {1.0f, 0.0f, -1.0f, 0.0f},
    {0.0f, 1.0f, 1.0f, 0.0f},
    {0.0f, -1.0f, 1.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, -1.0f}
};

__constant__ float B[4][4] = {
    {1.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, -1.0f, 1.0f},
    {-1.0f, 1.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, -1.0f}
};

__constant__ float A_T[2][4] = {
    {1.0f, 1.0f, 1.0f, 0.0f},
    {0.0f, 1.0f, -1.0f, -1.0f}
};

__constant__ float A[4][2] = {
    {1.0f, 0.0f},
    {1.0f, 1.0f},
    {1.0f, -1.0f},
    {0.0f, -1.0f}
};

// --- Kernel 1: 滤波器变换 (Grid-Stride Loop) ---
// 这个内核处理的数据量不大，Grid-Stride循环已经足够高效
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
// 定义线程块处理的“宏观图块”的尺寸
constexpr int TILE_W = 32;
constexpr int TILE_H = 32;
// 为这个宏观图块计算所需的输入数据片（Patch）的尺寸
constexpr int PATCH_W = TILE_W * 2 + 2; // 16*2 + 2 = 34
constexpr int PATCH_H = TILE_H * 2 + 2; // 8*2 + 2 = 18

__global__ void transform_input_kernel_tiled(const float* __restrict__ image, float* __restrict__ V_padded, int N, int C, int H, int W, int P, int outH, int outW, int padded_C, int padded_P) {
    // 为输入数据片声明共享内存
    __shared__ float patch[PATCH_H][PATCH_W];

    // 1. 计算线程块和线程的全局任务
    // 每个线程块负责处理一个 TILE_W x TILE_H 的宏观图块
    const int tile_x_base = blockIdx.x * TILE_W;
    const int tile_y_base = blockIdx.y * TILE_H;
    const int nc = blockIdx.z;
    const int n = nc / C;
    const int c = nc % C;

    // 块内每个线程负责处理宏观图块中的一个小图块
    const int thx = threadIdx.x; // 块内 x 坐标 (0..TILE_W-1)
    const int thy = threadIdx.y; // 块内 y 坐标 (0..TILE_H-1)

    // 2. 协作加载大尺寸数据片到共享内存
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
    __syncthreads(); // 确保所有数据已加载

    // 3. 从共享内存中计算
    // 计算当前线程负责的图块在输出和输入中的全局坐标
    const int tile_x = tile_x_base + thx;
    const int tile_y = tile_y_base + thy;
    const int tiles_w = (outW + 1) / 2;

    if (tile_x < tiles_w && tile_y < (outH + 1) / 2) {
        // 从共享内存中读取当前线程所需的4x4数据
        float d[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                d[i][j] = patch[thy * 2 + i][thx * 2 + j];
            }
        }

        // 执行变换
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

        // 4. 将结果写回全局内存
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


// --- 主机函数: Winograd 卷积调度器 ---
void winograd_conv(thrust::device_vector<float>& image,
                   thrust::device_vector<float>& filter, 
                   thrust::device_vector<float>& out,
                   thrust::device_vector<float>& U, // Unused
                   thrust::device_vector<float>& V, // Unused
                   thrust::device_vector<float>& M, // Unused
                   int H, int W, int C, int K, int N) {
    const int outH = H - 2;
    const int outW = W - 2;

    if (outH <= 0 || outW <= 0) {
        cudaMemset(out.data().get(), 0, out.size() * sizeof(float));
        return;
    }
    
    const int P = N * ((outH + 1) / 2) * ((outW + 1) / 2);
    if (P == 0) {
        cudaMemset(out.data().get(), 0, out.size() * sizeof(float));
        return;
    }

    int padded_C = (C + 15) & ~15;
    int padded_K = (K + 15) & ~15;
    int padded_P = (P + 15) & ~15;

    thrust::device_vector<float> U_padded(16 * padded_K * padded_C);
    thrust::device_vector<float> V_padded(16 * padded_C * padded_P);
    thrust::device_vector<float> M_padded(16 * padded_K * padded_P);
    
    cudaMemset(U_padded.data().get(), 0, U_padded.size() * sizeof(float));
    cudaMemset(V_padded.data().get(), 0, V_padded.size() * sizeof(float));
    cudaMemset(M_padded.data().get(), 0, M_padded.size() * sizeof(float));
    
    // --- 优化后的内核启动 ---
    const int threads_per_block = 256;

    // 1. 滤波器变换 (Grid-Stride)
    const int total_filters = K * C;
    int grid_size_filter = (total_filters + threads_per_block - 1) / threads_per_block;
    transform_filter_kernel_merged<<<grid_size_filter, threads_per_block>>>(filter.data().get(), U_padded.data().get(), C, K, padded_C, padded_K);

    // 2. 输入变换 (高级Tiling)
    dim3 block_dim_input(TILE_W, TILE_H); // e.g., 16x8 = 128 threads
    const int num_tiles_w = (outW + 1) / 2;
    const int num_tiles_h = (outH + 1) / 2;
    dim3 grid_dim_input(
        (num_tiles_w + TILE_W - 1) / TILE_W,
        (num_tiles_h + TILE_H - 1) / TILE_H,
        N * C // 每个 (n,c) 对都有一个2D的线程块网格
    );
    transform_input_kernel_tiled<<<grid_dim_input, block_dim_input>>>(image.data().get(), V_padded.data().get(), N, C, H, W, P, outH, outW, padded_C, padded_P);

    // 3. GEMM 计算 (调用 cuBLAS)
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int batch_count = 16;

    const float** h_U_array = new const float*[batch_count];
    const float** h_V_array = new const float*[batch_count];
    float** h_M_array = new float*[batch_count];

    for(int i = 0; i < batch_count; ++i) {
        h_U_array[i] = U_padded.data().get() + i * padded_K * padded_C;
        h_V_array[i] = V_padded.data().get() + i * padded_C * padded_P;
        h_M_array[i] = M_padded.data().get() + i * padded_K * padded_P;
    }

    const float **d_U_array, **d_V_array;
    float **d_M_array;
    cudaMalloc(&d_U_array, batch_count * sizeof(float*));
    cudaMalloc(&d_V_array, batch_count * sizeof(float*));
    cudaMalloc(&d_M_array, batch_count * sizeof(float*));
    cudaMemcpy(d_U_array, h_U_array, batch_count * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_array, h_V_array, batch_count * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M_array, h_M_array, batch_count * sizeof(float*), cudaMemcpyHostToDevice);

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

    // 4. 输出变换 (Grid-Stride)
    const int total_output_tiles = K * P;
    int grid_size_output = (total_output_tiles + threads_per_block - 1) / threads_per_block;
    transform_output_kernel<<<grid_size_output, threads_per_block>>>(M_padded.data().get(), out.data().get(), N, K, P, outH, outW, padded_K, padded_P);

    cudaDeviceSynchronize();
}

