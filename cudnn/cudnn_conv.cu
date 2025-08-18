#include "winograd.cuh"
#include <cudnn.h>
#include <iostream>
#include <vector>

// Macro to check cuDNN API call status
#define CUDNN_CHECK(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error in " << __FILE__ << ":" << __LINE__ \
                  << " : " << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

void cudnn_winograd_conv(thrust::device_vector<float>& image,
                         thrust::device_vector<float>& filter, 
                         thrust::device_vector<float>& out,
                         int H, int W, int C, int K, int N) {

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    // 1. Define Tensor Descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, 3, 3));
    
    const int outH = H - 2;
    const int outW = W - 2;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, outH, outW));

    // 2. Define Convolution Descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // 3. Directly force the Winograd algorithm, bypassing search
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    
    // 4. Get workspace size for the forced algorithm
    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_bytes));

    void* d_workspace = nullptr;
    if (workspace_bytes > 0) {
        cudaMalloc(&d_workspace, workspace_bytes);
    }

    // 5. Attempt to execute the convolution
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cudnnStatus_t status = cudnnConvolutionForward(handle,
                                      &alpha,
                                      input_desc, image.data().get(),
                                      filter_desc, filter.data().get(),
                                      conv_desc,
                                      algo,
                                      d_workspace, workspace_bytes,
                                      &beta,
                                      output_desc, out.data().get());
    
    // If the forced algorithm fails, print an error and zero out the result
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cuDNN error: Forced WINOGRAD algorithm failed for this layer with status: " 
                  << cudnnGetErrorString(status) << std::endl;
        cudaMemset(out.data().get(), 0, out.size() * sizeof(float));
    }

    // 6. Clean up resources
    if (d_workspace) {
        cudaFree(d_workspace);
    }
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);
}

