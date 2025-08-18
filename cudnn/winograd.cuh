#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

// Naive baseline
void naive_conv(thrust::device_vector<float>& image,
                thrust::device_vector<float>& filter, 
                thrust::device_vector<float>& out,
                int H, int W, int C, int K, int N);

// Our custom Winograd implementation
void winograd_conv(thrust::device_vector<float>& image, 
                   thrust::device_vector<float>& filter, 
                   thrust::device_vector<float>& out,
                   int H, int W, int C, int K, int N);

// Official cuDNN Winograd implementation
void cudnn_winograd_conv(thrust::device_vector<float>& image, 
                         thrust::device_vector<float>& filter, 
                         thrust::device_vector<float>& out,
                         int H, int W, int C, int K, int N);

