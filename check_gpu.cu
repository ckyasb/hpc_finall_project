#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-enabled devices found." << std::endl;
        return 1;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, dev);

        std::cout << "--- GPU 设备 #" << dev << " ---" << std::endl;
        std::cout << "型号: " << props.name << std::endl;
        std::cout << "计算能力: " << props.major << "." << props.minor << std::endl;

        if (props.major >= 8) {
            std::cout << "状态: 支持 TF32" << std::endl;
        } else {
            std::cout << "状态: 不支持 TF32" << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}

