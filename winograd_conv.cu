#include "winograd.cuh"
#include <thrust/device_vector.h>
#include <cstdio> 
#include <cstdlib>
#include <cublas_v2.h> // 引入 cuBLAS 库

// 定义矩阵乘法Tile的维度
constexpr int TILE_DIM = 16;

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

// --- Kernel 1: 滤波器变换 (输出 float) ---
__global__ void transform_filter_kernel_fp32(const float* __restrict__ filter, float* __restrict__ U_padded, int C, int K, int padded_C, int padded_K) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (k >= K || c >= C) return;

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

// --- Kernel 2: 输入变换 (输出 float) ---
__global__ void transform_input_kernel_fp32(const float* __restrict__ image, float* __restrict__ V_padded, int N, int C, int H, int W, int P, int outH, int outW, int padded_C, int padded_P) {
    int p_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (p_idx >= P || c >= C) return;

    const int tiles_w = (outW + 1) / 2;
    const int tiles_per_image = ((outH + 1) / 2) * tiles_w;
    const int n = p_idx / tiles_per_image;
    const int p_local = p_idx % tiles_per_image;
    const int tile_y = p_local / tiles_w;
    const int tile_x = p_local % tiles_w;
    
    int h_start = tile_y * 2;
    int w_start = tile_x * 2;

    float d[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int cur_h = h_start + i;
            int cur_w = w_start + j;
            if (cur_h < H && cur_w < W) {
                d[i][j] = image[(n * C + c) * H * W + cur_h * W + cur_w];
            } else {
                d[i][j] = 0.0f;
            }
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

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int gemm_idx = i * 4 + j;
            V_padded[gemm_idx * padded_C * padded_P + c * padded_P + p_idx] = v_ncp[i][j];
        }
    }
}

// --- Kernel 4: 输出变换 ---
__global__ void transform_output_kernel(const float* __restrict__ M_padded, float* __restrict__ output, int N, int K, int P, int outH, int outW, int padded_K, int padded_P) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int p_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (k >= K || p_idx >= P) return;

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

    int padded_C = (C + TILE_DIM - 1) & ~(TILE_DIM - 1);
    int padded_K = (K + TILE_DIM - 1) & ~(TILE_DIM - 1);
    int padded_P = (P + TILE_DIM - 1) & ~(TILE_DIM - 1);

    thrust::device_vector<float> U_padded(16 * padded_K * padded_C);
    thrust::device_vector<float> V_padded(16 * padded_C * padded_P);
    thrust::device_vector<float> M_padded(16 * padded_K * padded_P);
    
    cudaMemset(U_padded.data().get(), 0, U_padded.size() * sizeof(float));
    cudaMemset(V_padded.data().get(), 0, V_padded.size() * sizeof(float));
    cudaMemset(M_padded.data().get(), 0, M_padded.size() * sizeof(float));

    dim3 block_dim_2d(16, 16);
    
    // 1. 滤波器变换
    dim3 grid_dim_filter((K + block_dim_2d.x - 1) / block_dim_2d.x, (C + block_dim_2d.y - 1) / block_dim_2d.y);
    transform_filter_kernel_fp32<<<grid_dim_filter, block_dim_2d>>>(filter.data().get(), U_padded.data().get(), C, K, padded_C, padded_K);

    // 2. 输入变换
    dim3 grid_dim_input((P + block_dim_2d.x - 1) / block_dim_2d.x, (C + block_dim_2d.y - 1) / block_dim_2d.y);
    transform_input_kernel_fp32<<<grid_dim_input, block_dim_2d>>>(image.data().get(), V_padded.data().get(), N, C, H, W, P, outH, outW, padded_C, padded_P);

    // 3. GEMM 计算 (调用 cuBLAS)
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int batch_count = 16;

    // 为指针数组分配主机内存
    const float** h_U_array = new const float*[batch_count];
    const float** h_V_array = new const float*[batch_count];
    float** h_M_array = new float*[batch_count];

    // 填充主机指针数组
    for(int i = 0; i < batch_count; ++i) {
        h_U_array[i] = U_padded.data().get() + i * padded_K * padded_C;
        h_V_array[i] = V_padded.data().get() + i * padded_C * padded_P;
        h_M_array[i] = M_padded.data().get() + i * padded_K * padded_P;
    }

    // 为指针数组分配设备内存并从主机复制
    const float **d_U_array, **d_V_array;
    float **d_M_array;
    cudaMalloc(&d_U_array, batch_count * sizeof(float*));
    cudaMalloc(&d_V_array, batch_count * sizeof(float*));
    cudaMalloc(&d_M_array, batch_count * sizeof(float*));
    cudaMemcpy(d_U_array, h_U_array, batch_count * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_array, h_V_array, batch_count * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M_array, h_M_array, batch_count * sizeof(float*), cudaMemcpyHostToDevice);

    // BUG FIX: Correctly call cuBLAS for row-major matrices.
    // To compute C = A * B (row-major), we call cuBLAS with C = B * A and swapped dimensions.
    // Our A is U(K,C), B is V(C,P). We want C=M(K,P).
    // So, we call cublas with m=P, n=K, k=C.
    // The first matrix passed is V, the second is U.
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       padded_P, padded_K, padded_C,
                       &alpha,
                       d_V_array, padded_P,  // A for cuBLAS is our V
                       d_U_array, padded_C,  // B for cuBLAS is our U
                       &beta,
                       d_M_array, padded_P,  // C for cuBLAS is our M
                       batch_count);

    // 清理资源
    delete[] h_U_array;
    delete[] h_V_array;
    delete[] h_M_array;
    cudaFree(d_U_array);
    cudaFree(d_V_array);
    cudaFree(d_M_array);
    cublasDestroy(handle);

    // 4. 输出变换
    dim3 grid_dim_output((K + block_dim_2d.x - 1) / block_dim_2d.x, (P + block_dim_2d.y - 1) / block_dim_2d.y);
    transform_output_kernel<<<grid_dim_output, block_dim_2d>>>(M_padded.data().get(), out.data().get(), N, K, P, outH, outW, padded_K, padded_P);

    cudaDeviceSynchronize();
}

