#include <cudnn_v9.h>
#include <iostream>
#include "../../include/kernels.h"
#include "cublasLt.h"
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "math.h"
#include "stdio.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "../helper/helper.h"

static cudnnHandle_t cudnn_handle;

void* create_pooling_desc() {
    PoolDesc* desc = new PoolDesc();
    cudnnCreateTensorDescriptor(&desc->input_desc);
    cudnnCreateTensorDescriptor(&desc->output_desc);
    cudnnCreatePoolingDescriptor(&desc->pooling_desc);
    return (void*)desc;
}

void setup_pooling_descriptor(std::vector<int>& kernels, std::vector<int>& paddings,
                              std::vector<int>& strides, std::vector<int>& in_shape,
                              std::vector<int>& out_shape, std::string optype, std::string dtype,
                              std::string layout, void* desc) {
    PoolDesc* desc_ = (PoolDesc*)desc;
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

    cudnnPoolingMode_t mode;
    if (optype == "GlobalAveragePool") {
        mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        cudnnNanPropagation_t maxpoolingNanOpt = CUDNN_PROPAGATE_NAN;
        cudnnSetPooling2dDescriptor(desc_->pooling_desc, mode, maxpoolingNanOpt, in_shape.at(2),
                                    in_shape.at(3), 0, 0, 1, 1);
    } else if (optype == "MaxPool") {
        mode = CUDNN_POOLING_MAX;
        cudnnNanPropagation_t maxpoolingNanOpt = CUDNN_PROPAGATE_NAN;
        cudnnSetPooling2dDescriptor(desc_->pooling_desc, mode, maxpoolingNanOpt, kernels.at(0),
                                    kernels.at(1), paddings.at(0), paddings.at(1), strides.at(0),
                                    strides.at(1));
    }
}


bool pooling_cudnn_backend(int64_t in_ptr, int64_t out_ptr, std::vector<int> kernels,
                     std::vector<int> paddings, std::vector<int> strides, std::vector<int> in_shape,
                     std::vector<int> out_shape, std::string optype, std::string dtype,
                     std::string layout, int64_t pstream, void* desc) {

    // setup_pooling_descriptor(kernels, paddings, strides, in_shape, out_shape, optype, dtype, layout, desc);

    PoolDesc* desc_ = (PoolDesc*)desc;
    if (!cudnn_handle) {
        cudnnCreate(&cudnn_handle);
        cudnnSetStream(cudnn_handle, (cudaStream_t)pstream);
    }

    float alpha_ = 1.0f;
    float beta_ = 0.0f;

    cudnnStatus_t Error =
        cudnnPoolingForward(cudnn_handle, desc_->pooling_desc, &alpha_, desc_->input_desc,
                            (void*)in_ptr, &beta_, desc_->output_desc, (void*)out_ptr);

    if (Error != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "[Error] cudnn forward failed!\n");
    }
    return true;
}
