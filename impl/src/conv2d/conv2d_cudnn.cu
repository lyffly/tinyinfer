#include "../../include/kernels.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cublasLt.h"
#include "cublas_v2.h"
#include "cudnn.h"
#include "stdio.h"
#include "math.h"
#include <cudnn_ops_infer.h>
#include <iostream>

#include <cstdint>
#include <vector>
#include <string>
#include <cstdio>
#include "../helper/helper.h"

static cudnnHandle_t cudnn_handle;

void setup_descriptor_nchw(std::vector<int> &kernels, std::vector<int> &paddings, std::vector<int> &strides, std::vector<int> &dilations, int group,
                std::vector<int> &in_shape, std::vector<int> &weight_shape, std::vector<int> &bias_shape, std::vector<int> &out_shape, 
                std::string dtype, std::string layout, cudnnTensorDescriptor_t &input_desc,
                cudnnTensorDescriptor_t &output_desc, cudnnFilterDescriptor_t &kernel_desc, cudnnConvolutionDescriptor_t &conv_desc) {
    int batch = in_shape.at(0);
    int inc = in_shape.at(1);
    int inh = in_shape.at(2);
    int inw = in_shape.at(3);

    int outc = out_shape.at(1);
    int outh = out_shape.at(2);
    int outw = out_shape.at(3);

    if (!cudnn_handle) cudnnCreate(&cudnn_handle);

	cudnnCreateTensorDescriptor(&input_desc);
    if (dtype=="float16") {
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
            batch, inc, inh, inw);
    } else if (dtype=="float32") {
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            batch, inc, inh, inw);
    }

	cudnnCreateTensorDescriptor(&output_desc);
    if (dtype=="float16") {
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
            batch, outc, outh, outw);
    } else if (dtype=="float32") {
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            batch, outc, outh, outw);
    }

	cudnnCreateFilterDescriptor(&kernel_desc);
    if (dtype=="float16") {
        cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
	        outc, inc, kernels.at(0), kernels.at(1));
    } else if (dtype=="float32") {
        cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
	        outc, inc, kernels.at(0), kernels.at(1));
    }

	cudnnCreateConvolutionDescriptor(&conv_desc);
    if (dtype=="float16") {
        cudnnSetConvolution2dDescriptor(conv_desc, paddings.at(0), paddings.at(1),
            strides.at(0), strides.at(1),
            dilations.at(0), dilations.at(1), 
            CUDNN_CONVOLUTION, CUDNN_DATA_HALF);
    } else if (dtype=="float32") {
        cudnnSetConvolution2dDescriptor(conv_desc, paddings.at(0), paddings.at(1),
            strides.at(0), strides.at(1),
            dilations.at(0), dilations.at(1), 
            CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
    }
    cudnnSetConvolutionGroupCount(conv_desc, group);
    cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
    cudnnGetConvolutionMathType(conv_desc, &math_type);

}

int64_t get_conv2d_algo(std::vector<int> kernels, std::vector<int> paddings, std::vector<int> strides, 
                std::vector<int> dilations, int group,
                std::vector<int> in_shape, std::vector<int> weight_shape, std::vector<int> bias_shape, std::vector<int> out_shape, 
                std::string dtype, std::string layout) {
    if (!cudnn_handle) cudnnCreate(&cudnn_handle);

    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t kernel_desc;
    cudnnConvolutionDescriptor_t conv_desc;
	setup_descriptor_nchw(kernels, paddings, strides, dilations, group, in_shape, weight_shape, bias_shape, out_shape, 
                dtype, layout,input_desc,output_desc, kernel_desc, conv_desc);

    cudnnConvolutionFwdAlgoPerf_t perfResults[10];
    int returnedAlgoCount = 0;
    cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn_handle,
        input_desc,
        kernel_desc,
        conv_desc,
        output_desc,
        10,
        &returnedAlgoCount,
        perfResults);
    if (returnedAlgoCount>0) {
        return int64_t(perfResults[0].algo);
    } else {
        return int64_t(-1);
    }
}

int64_t get_conv2d_workspace_size(std::vector<int> kernels, std::vector<int> paddings, std::vector<int> strides, 
                std::vector<int> dilations, int group,
                std::vector<int> in_shape, std::vector<int> weight_shape, std::vector<int> bias_shape, std::vector<int> out_shape, 
                std::string dtype, std::string layout, int64_t algo) {
    if (!cudnn_handle) cudnnCreate(&cudnn_handle);

    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t kernel_desc;
    cudnnConvolutionDescriptor_t conv_desc;
	setup_descriptor_nchw(kernels, paddings, strides, dilations, group, in_shape, weight_shape, bias_shape, out_shape, 
                dtype, layout,input_desc,output_desc, kernel_desc, conv_desc);
	
    size_t space_size = 0;
    cudnnConvolutionFwdAlgo_t algo_ = (cudnnConvolutionFwdAlgo_t)algo;
    
    cudnnStatus_t Error = cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, input_desc,kernel_desc, conv_desc, output_desc, 
        algo_, &space_size);
    if (Error != CUDNN_STATUS_SUCCESS) {
		fprintf(stderr, "[Error] cudnn get workspace size failed!\n");
	}

    return int64_t(space_size);
}


bool conv2d_backend(int64_t in_ptr, int64_t weight_ptr, int64_t bias_ptr, int64_t out_ptr, int64_t workspace_size, 
                int64_t workspace_ptr, int64_t algo, std::vector<int> kernels, std::vector<int> paddings, std::vector<int> strides, 
                std::vector<int> dilations, int group,
                std::vector<int> in_shape, std::vector<int> weight_shape, std::vector<int> bias_shape, std::vector<int> out_shape, 
                std::string dtype, std::string layout, int64_t pstream) {

    if (!cudnn_handle) cudnnCreate(&cudnn_handle);

    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t kernel_desc;
    cudnnConvolutionDescriptor_t conv_desc;
	setup_descriptor_nchw(kernels, paddings, strides, dilations, group, in_shape, weight_shape, bias_shape, out_shape, 
                dtype, layout,input_desc,output_desc, kernel_desc, conv_desc);


    size_t space_size_ = (size_t)workspace_size;
    cudnnConvolutionFwdAlgo_t algo_ = (cudnnConvolutionFwdAlgo_t)algo;

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    cudnnStatus_t Error = cudnnConvolutionForward(cudnn_handle, 
		&alpha_, input_desc,
		(void*)in_ptr, kernel_desc,
        (void*)weight_ptr, conv_desc,
		algo_, (void*)workspace_ptr,
		space_size_, &beta_,
		output_desc, (void*)out_ptr);
    // todo add bias


	if (Error != CUDNN_STATUS_SUCCESS) {
		fprintf(stderr, "[Error] cudnn forward failed!\n");
	}

    return true;
}