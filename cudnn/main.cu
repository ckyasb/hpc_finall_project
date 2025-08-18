#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <iomanip>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>

#include "winograd.cuh"

#define LOOP_NUM 3
#define BLUE "\033[34m"
#define RESET "\033[0m"

class Config {
public:
    std::vector<int> C, H, W, K, Batch;
    int layer_num;

    Config(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open config file: " + filename);
        }
        
        file >> layer_num;
        C.resize(layer_num);
        H.resize(layer_num);
        W.resize(layer_num);
        K.resize(layer_num);
        Batch.resize(layer_num);
        
        for (int i = 0; i < layer_num; i++) {
            file >> C[i] >> H[i] >> W[i] >> K[i] >> Batch[i];
        }
    }
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    auto cfg = std::make_unique<Config>(argv[1]);
    
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    float milliseconds = 0;

    // --- 数据准备 ---
    thrust::host_vector<thrust::host_vector<float>> h_images(cfg->layer_num);
    thrust::host_vector<thrust::host_vector<float>> h_filters(cfg->layer_num);

    std::random_device rd;
    thrust::default_random_engine rng(rd());
    thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int l = 0; l < cfg->layer_num; l++) {
        int H = cfg->H[l], W = cfg->W[l], C = cfg->C[l], K = cfg->K[l], N = cfg->Batch[l];
        long sizeI = H * W, sizeF = 9;
        h_images[l].resize(N * C * sizeI);
        h_filters[l].resize(K * C * sizeF);
        thrust::generate(h_images[l].begin(), h_images[l].end(), [&] { return dist(rng); });
        thrust::generate(h_filters[l].begin(), h_filters[l].end(), [&] { return dist(rng); });
    }

    // --- 运行官方cuDNN Winograd卷积 ---
    std::cout << "\n=== Running Official cuDNN Winograd ===" << std::endl;
    double cudnn_total_time = 0;
    long total_flops = 0;

    for (int l = 0; l < cfg->layer_num; l++) {
        int H = cfg->H[l], W = cfg->W[l], C = cfg->C[l], K = cfg->K[l], N = cfg->Batch[l];
        if (H < 3 || W < 3) {
            std::cout << "Layer " << std::setw(2) << l << ": SKIPPED (input too small)" << std::endl;
            continue;
        }
        long sizeO = (H-2) * (W-2);
        
        thrust::device_vector<float> d_image = h_images[l];
        thrust::device_vector<float> d_filter = h_filters[l];
        thrust::device_vector<float> d_out(N * K * sizeO);

        // 预热
        cudnn_winograd_conv(d_image, d_filter, d_out, H, W, C, K, N);
        
        // 计时
        cudaEventRecord(start_event);
        for (int i = 0; i < LOOP_NUM; i++) {
            cudnn_winograd_conv(d_image, d_filter, d_out, H, W, C, K, N);
        }
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        double cudnn_time = static_cast<double>(milliseconds) / LOOP_NUM;
        
        long flops = static_cast<long>(N) * K * C * (H-2) * (W-2) * 18;
        double cudnn_gflops = flops * 1e-6 / cudnn_time;
        
        std::cout << "Layer " << std::setw(2) << l << ": " << std::fixed << std::setprecision(2)
                  << cudnn_time << " ms (" << cudnn_gflops << " GFLOPS)" << std::endl;
        
        cudnn_total_time += cudnn_time;
        total_flops += flops;
    }
    
    std::cout << "\n=== Final Results ===" << std::endl;
    std::cout << "cuDNN Total: " << BLUE << std::fixed << std::setprecision(2) << cudnn_total_time 
              << " ms (" << total_flops * 1e-6 / cudnn_total_time << " GFLOPS)" << RESET << std::endl;
    
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    
    return 0;
}

