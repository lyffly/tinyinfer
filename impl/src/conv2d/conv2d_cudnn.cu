#include <cudnn_v9.h>
#include <iostream>
#include "../../include/kernels.h"
#include "cublasLt.h"
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "cudnn.h"
#include "math.h"
#include "stdio.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "../helper/helper.h"

static cudnnHandle_t cudnn_handle;

void* create_conv2d_desc() {
    ConvDesc* desc = new ConvDesc();
    cudnnCreateTensorDescriptor(&desc->input_desc);
    cudnnCreateTensorDescriptor(&desc->output_desc);
    cudnnCreateFilterDescriptor(&desc->kernel_desc);
    cudnnCreateTensorDescriptor(&desc->bias_desc);
    cudnnCreateConvolutionDescriptor(&desc->conv_desc);
    cudnnCreateActivationDescriptor(&desc->activation_desc);
    cudnnCreateActivationDescriptor(&desc->activation_desc);
    return (void*)desc;
}

void setup_descriptor(std::vector<int>& kernels, std::vector<int>& paddings,
                      std::vector<int>& strides, std::vector<int>& dilations, int group,
                      std::vector<int>& in_shape, std::vector<int>& weight_shape,
                      std::vector<int>& bias_shape, std::vector<int>& out_shape, std::string dtype,
                      std::string layout, void* desc) {
    ConvDesc* desc_ = (ConvDesc*)desc;
    cudnnDataType_t infer_data_type;
    cudnnTensorFormat_t infer_layout;
    if (layout == "nchw") {
        infer_layout = CUDNN_TENSOR_NCHW;
    } else if (layout == "nhwc") {
        infer_layout = CUDNN_TENSOR_NHWC;
    }
    if (dtype == "float16") {
        infer_data_type = CUDNN_DATA_HALF;
    } else if (dtype == "float32") {
        infer_data_type = CUDNN_DATA_FLOAT;
    }

    int batch = in_shape.at(0);
    int inc = in_shape.at(1);
    int inh = in_shape.at(2);
    int inw = in_shape.at(3);

    int outc = out_shape.at(1);
    int outh = out_shape.at(2);
    int outw = out_shape.at(3);
    // printf("input shape : %d %d %d %d \n", batch, inc, inh, inw);
    // printf("output shape : %d %d %d %d \n", batch, outc, outh, outw);

    cudnnSetTensor4dDescriptor(desc_->input_desc, infer_layout, infer_data_type, batch, inc, inh,
                               inw);
    cudnnSetTensor4dDescriptor(desc_->output_desc, infer_layout, infer_data_type, batch, outc, outh,
                               outw);
    cudnnSetFilter4dDescriptor(desc_->kernel_desc, infer_data_type, infer_layout, outc, inc,
                               kernels.at(0), kernels.at(1));

    // printf("kernel shape : %d %d %d %d \n", outc, inc, kernels.at(0), kernels.at(1));

    cudnnSetTensor4dDescriptor(desc_->bias_desc,
                               /*format=*/infer_layout,
                               /*dataType=*/infer_data_type,
                               /*batch_size=*/1,
                               /*channels=*/outc,
                               /*image_height=*/1,
                               /*image_width=*/1);
    // printf("bias shape : %d %d %d %d \n", 1, outc, 1, 1);

    cudnnSetConvolutionMathType(desc_->conv_desc, CUDNN_TENSOR_OP_MATH);

    cudnnSetConvolution2dDescriptor(desc_->conv_desc, paddings.at(0), paddings.at(1), strides.at(0),
                                    strides.at(1), dilations.at(0), dilations.at(1),
                                    CUDNN_CROSS_CORRELATION, infer_data_type);

    // printf("padding : %d %d, stride: %d %d, dilation: %d %d \n", paddings.at(0), paddings.at(1),
    //        strides.at(0), strides.at(1), dilations.at(0), dilations.at(1));
    cudnnSetConvolutionGroupCount(desc_->conv_desc, group);
    cudnnSetActivationDescriptor(desc_->activation_desc, CUDNN_ACTIVATION_IDENTITY,
                                 CUDNN_PROPAGATE_NAN, 0);
}

int64_t get_conv2d_algo(std::vector<int> kernels, std::vector<int> paddings,
                        std::vector<int> strides, std::vector<int> dilations, int group,
                        std::vector<int> in_shape, std::vector<int> weight_shape,
                        std::vector<int> bias_shape, std::vector<int> out_shape, std::string dtype,
                        std::string layout, int64_t pstream, void* desc) {
    ConvDesc* desc_ = (ConvDesc*)desc;
    if (!cudnn_handle) {
        cudnnCreate(&cudnn_handle);
        cudnnSetStream(cudnn_handle, (cudaStream_t)pstream);
    }

    setup_descriptor(kernels, paddings, strides, dilations, group, in_shape, weight_shape,
                     bias_shape, out_shape, dtype, layout, desc_);

    cudnnConvolutionFwdAlgoPerf_t perfResults[10];
    int returnedAlgoCount = 0;
    // cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle, input_desc, kernel_desc, conv_desc,
    //                                        output_desc, 1, &returnedAlgoCount, perfResults);
    cudnnFindConvolutionForwardAlgorithm(cudnn_handle, desc_->input_desc, desc_->kernel_desc,
                                         desc_->conv_desc, desc_->output_desc, 2,
                                         &returnedAlgoCount, perfResults);

    // printf("find algo: %d, math %d \n\n ", int32_t(perfResults[0].algo),
    //        int32_t(perfResults[0].mathType));

    // return int64_t(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
    if (returnedAlgoCount > 0) {
        return int64_t(perfResults[0].algo);
    } else {
        return int64_t(-1);
    }
}

int64_t get_conv2d_workspace_size(std::vector<int> kernels, std::vector<int> paddings,
                                  std::vector<int> strides, std::vector<int> dilations, int group,
                                  std::vector<int> in_shape, std::vector<int> weight_shape,
                                  std::vector<int> bias_shape, std::vector<int> out_shape,
                                  std::string dtype, std::string layout, int64_t algo,
                                  int64_t pstream, void* desc) {
    ConvDesc* desc_ = (ConvDesc*)desc;
    if (!cudnn_handle) {
        cudnnCreate(&cudnn_handle);
        cudnnSetStream(cudnn_handle, (cudaStream_t)pstream);
    }

    size_t space_size = 0;
    cudnnConvolutionFwdAlgo_t algo_ = (cudnnConvolutionFwdAlgo_t)algo;

    cudnnStatus_t Error = cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_handle, desc_->input_desc, desc_->kernel_desc, desc_->conv_desc, desc_->output_desc,
        algo_, &space_size);
    if (Error != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "[Error] cudnn get workspace size failed!\n");
    }

    return int64_t(space_size);
}

// batch*c*hw + 1*c*1 = batch*c*hw
template <typename T>
__global__ void add_conv2d_bias_fp(T* inout, T* bias, int batch, int c, int hw, int length) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    auto inout_ptr = reinterpret_cast<T*>(inout);
    auto bias_ptr = reinterpret_cast<T*>(bias);
    if ((index < length)) {
        int index_bias = index % (c * hw) / hw;
        inout_ptr[index] = inout_ptr[index] + bias_ptr[index_bias];
    }
}

bool conv2d_backend(int64_t in_ptr, int64_t weight_ptr, int64_t bias_ptr, int64_t out_ptr,
                    int64_t workspace_size, int64_t workspace_ptr, int64_t algo,
                    std::vector<int> kernels, std::vector<int> paddings, std::vector<int> strides,
                    std::vector<int> dilations, int group, std::vector<int> in_shape,
                    std::vector<int> weight_shape, std::vector<int> bias_shape,
                    std::vector<int> out_shape, std::string dtype, std::string layout,
                    int64_t pstream, void* desc) {
    ConvDesc* desc_ = (ConvDesc*)desc;
    if (!cudnn_handle) {
        cudnnCreate(&cudnn_handle);
        cudnnSetStream(cudnn_handle, (cudaStream_t)pstream);
    }

    size_t space_size_ = (size_t)workspace_size;
    cudnnConvolutionFwdAlgo_t algo_ = (cudnnConvolutionFwdAlgo_t)algo;

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    // conv
    // {
    //     // cudnnStatus_t Error = cudnnConvolutionForward(cudnn_handle,
    //     cudnnConvolutionForward(cudnn_handle, &alpha_, input_desc, (void*)in_ptr, kernel_desc,
    //                             (void*)weight_ptr, conv_desc, algo_, (void*)workspace_ptr,
    //                             space_size_, &beta_, output_desc, (void*)out_ptr);
    //     // if (Error != CUDNN_STATUS_SUCCESS) {
    //     //   fprintf(stderr, "[Error] cudnn forward failed!\n");
    //     // }

    //     // add bias
    //     int block_size = 512;
    //     int batch = out_shape.at(0);
    //     int outc = out_shape.at(1);
    //     int hw = out_shape.at(2) * out_shape.at(3);
    //     int length = batch * outc * hw;
    //     cudaStream_t stream = (cudaStream_t)pstream;
    //     int grid_size = (length + block_size - 1) / block_size;
    //     if (dtype == "float32") {
    //         add_conv2d_bias_fp<float><<<grid_size, block_size, 0, stream>>>(
    //             (float*)out_ptr, (float*)bias_ptr, batch, outc, hw, length);
    //     } else if (dtype == "float16") {
    //         add_conv2d_bias_fp<half><<<grid_size, block_size, 0, stream>>>(
    //             (half*)out_ptr, (half*)bias_ptr, batch, outc, hw, length);
    //     }
    // }

    // conv bias activation
    {
        // printf("[conv] algo %d, space size %ld, ptr %p \n\n", (int)algo_, space_size_,
        //        (void*)workspace_ptr);

        cudnnStatus_t Error = cudnnConvolutionBiasActivationForward(
            cudnn_handle, &alpha_, desc_->input_desc, (void*)in_ptr, desc_->kernel_desc,
            (void*)weight_ptr, desc_->conv_desc, algo_, (void*)workspace_ptr, space_size_, &beta_,
            desc_->output_desc, (void*)out_ptr, desc_->bias_desc, (void*)bias_ptr,
            desc_->activation_desc, desc_->output_desc, (void*)out_ptr);

        if (Error != CUDNN_STATUS_SUCCESS) {
            fprintf(stderr, "[Error] cudnn forward failed!\n");
        }
    }
    return true;
}
