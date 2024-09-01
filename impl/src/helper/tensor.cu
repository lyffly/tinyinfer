#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <system_error>
#include "../../include/kernels.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "helper.h"
#include "math.h"
#include "stdio.h"

int64_t GetSizeofDtype(DataType data_type) {
    if (data_type == DataType::INT64) {
        return 8;
    } else if (data_type == DataType::INT32 || data_type == DataType::FLOAT32) {
        return 4;
    } else if (data_type == DataType::FLOAT16 || data_type == DataType::HALF) {
        return 2;
    } else if (data_type == DataType::INT8 || data_type == DataType::BOOL) {
        return 1;
    }
    return 1;
}

int64_t GetProdofVector(std::vector<int64_t> shapes) {
    size_t sum = 1;
    for (auto& shape : shapes) {
        sum *= shape;
    }
    return sum;
}

void convert_fp32_to_fp16_cpu(float* in_ptr, half* out_ptr, int length) {
    for(auto i=0; i< length; i++) {
        out_ptr[i] = __float2half(in_ptr[i]);
    }
}

void convert_fp16_to_fp32_cpu(half* in_ptr, float* out_ptr, int length) {
    for(auto i=0; i< length; i++) {
        out_ptr[i] = __half2float(in_ptr[i]);
    }
}

__global__ void convert_fp32_to_fp16_cuda(float* in_ptr, half* out_ptr, int length) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length) {
        out_ptr[index] = __float2half(in_ptr[index]);
    }
}

__global__ void convert_fp16_to_fp32_cuda(half* in_ptr, float* out_ptr, int length) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length) {
        out_ptr[index] = __half2float(in_ptr[index]);
    }
}

YTensor::YTensor() {}

YTensor::~YTensor() {
    Free();
}

bool YTensor::Malloc(std::vector<int64_t> shape, DataType data_type, DataLayout layout) {
    this->Zeros(shape, data_type, layout);
    return true;
}

bool YTensor::Free() {
    if (this->cpu_ptr) {
        free(this->cpu_ptr);
        this->cpu_ptr = nullptr;
    }
    if (this->gpu_ptr) {
        cudaFree(this->gpu_ptr);
        this->gpu_ptr = nullptr;
    }
    this->data = nullptr;
    return true;
}

bool YTensor::ZerosLike(YTensor &ytensor) {
    return this->Zeros(ytensor.shape, ytensor.data_type, ytensor.layout);
}

bool YTensor::Zeros(std::vector<int64_t> shape, DataType data_type, DataLayout layout) {
    this->sizeoftype = GetSizeofDtype(data_type);
    this->length = GetProdofVector(shape);
    this->cpu_ptr = malloc(this->sizeoftype * this->length);
    if (not this->cpu_ptr) {
        std::cout << "malloc error \n";
    }
    memset(this->cpu_ptr, 0, this->sizeoftype * this->length);
    this->data = cpu_ptr;
    this->gpu_ptr = nullptr;
    this->rank = shape.size();
    this->shape = shape;
    this->data_type = data_type;
    this->layout = layout;
    this->is_gpu = false;
    this->tensor_type = TensorType::VARIABLE;
    this->data_len = this->sizeoftype * this->length;
    return true;
}

bool YTensor::Float() {
    if (this->data_type == DataType::FLOAT32) {
        return true;
    } else if ((!this->is_gpu) and this->data_type == DataType::HALF) {
        this->data_type = DataType::FLOAT32;
        this->sizeoftype = sizeof(float);
        float* tmp = (float*)malloc(this->length * sizeof(float));
        convert_fp16_to_fp32_cpu((half*)this->cpu_ptr, (float*)tmp, this->length);
        free(this->cpu_ptr);
        this->cpu_ptr = tmp;
        this->data = this->cpu_ptr;
        this->data_len = this->sizeoftype * this->length;
    } else if (this->is_gpu and this->data_type == DataType::HALF) {
        this->data_type = DataType::FLOAT32;
        this->sizeoftype = sizeof(float);
        int block_size = 512;
        int grid_size = (this->length + block_size - 1) / block_size;
        void* tmp;
        cudaMalloc((void**)&tmp, this->length * sizeof(half));
        cudaMemcpy(tmp, this->gpu_ptr, this->length * sizeof(half), cudaMemcpyDeviceToDevice);
        cudaFree(this->gpu_ptr);
        cudaMalloc((void**)&(this->gpu_ptr), this->length * this->sizeoftype);
        convert_fp16_to_fp32_cuda<<<grid_size, block_size>>>((half*)tmp, (float*)this->gpu_ptr,
                                                             this->length);
        this->data = this->gpu_ptr;
        this->data_len = this->sizeoftype * this->length;
        cudaFree(tmp);
    } else {
        printf("[Error] datatype not correct !!\n");
        return false;
    }
    return true;
}

bool YTensor::Half() {
     if (this->data_type == DataType::HALF) {
        return true;
    } else if ((!this->is_gpu) and this->data_type == DataType::FLOAT32) {
        this->data_type = DataType::FLOAT16;
        this->sizeoftype = sizeof(half);
        half* tmp = (half*)malloc(this->length * sizeof(half));
        convert_fp32_to_fp16_cpu((float*)this->cpu_ptr, (half*)tmp, this->length);
        free(this->cpu_ptr);
        this->cpu_ptr = tmp;
        this->data = this->cpu_ptr;
        this->data_len = this->sizeoftype * this->length;
    } else if (this->is_gpu and this->data_type == DataType::FLOAT32) {
        this->data_type = DataType::HALF;
        this->sizeoftype = sizeof(half);
        int block_size = 512;
        int grid_size = (this->length + block_size - 1) / block_size;
        void* tmp;
        cudaMalloc((void**)&tmp, this->length * sizeof(float));
        cudaMemcpy(tmp, this->gpu_ptr, this->length * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(this->gpu_ptr);
        cudaMalloc((void**)&(this->gpu_ptr), this->length * this->sizeoftype);
        convert_fp32_to_fp16_cuda<<<grid_size, block_size>>>((float*)tmp, (half*)this->gpu_ptr,
                                                             this->length);
        this->data = this->gpu_ptr;
        cudaFree(tmp);
        this->data_len = this->sizeoftype * this->length;
        return true;
    } else {
        printf("[Error] datatype not correct !!\n");
        return false;
    }
    return true;
}

bool YTensor::CUDA() {
    if (this->is_gpu) {
        checkCudaStatus(cudaMemcpy(this->gpu_ptr, this->cpu_ptr, this->sizeoftype * this->length,
                                   cudaMemcpyHostToDevice));
    } else {
        checkCudaStatus(cudaMalloc((void**)&(this->gpu_ptr), this->sizeoftype * this->length));
        checkCudaStatus(cudaMemcpy(this->gpu_ptr, this->cpu_ptr, this->sizeoftype * this->length,
                                   cudaMemcpyHostToDevice));
    }
    this->is_gpu = true;
    this->data = this->gpu_ptr;
    return true;
}

bool YTensor::CPU() {
    if (this->is_gpu) {
        checkCudaStatus(cudaMemcpy(this->cpu_ptr, this->gpu_ptr, this->sizeoftype * this->length,
                                   cudaMemcpyDeviceToHost));
        checkCudaStatus(cudaFree(this->gpu_ptr));
    }
    this->is_gpu = false;
    this->data = this->cpu_ptr;
    this->gpu_ptr = nullptr;
    return true;
}

bool YTensor::CopyNumpyData(int64_t ptr) {
    memcpy((void*)this->cpu_ptr, (void*)ptr, this->sizeoftype * this->length);
    return true;
}

void YTensor::SetDataPtr(int64_t ptr, bool is_gpu) {
    this->is_gpu = is_gpu;
    if (this->is_gpu) {
        this->data = (void*)ptr;
        this->gpu_ptr = (void*)ptr;
    } else {
        this->data = (void*)ptr;
        this->cpu_ptr = (void*)ptr;
    }
}

int64_t YTensor::GetDataPtr() {
    return (int64_t)(this->data);
}

void YTensor::SetShape(std::vector<int64_t> shape) {
    this->rank = shape.size();
    this->shape = shape;
    this->length = GetProdofVector(shape);
    this->data_len = this->sizeoftype * this->length;
}

std::vector<int64_t> YTensor::GetShape() {
    return this->shape;
}

bool YTensor::GetIsGPU() {
    return this->is_gpu;
}

void YTensor::SetIsGPU(bool is_gpu) {
    if ((!this->is_gpu) && is_gpu) {
        this->CUDA();
    }
}

DataType YTensor::GetDataType() {
    return this->data_type;
}

void YTensor::SetDataType(DataType type) {
    this->sizeoftype = GetSizeofDtype(type);
    this->data_type = type;
    this->data_len = this->sizeoftype * this->length;
}

DataLayout YTensor::GetDataLayout() {
    return this->layout;
}

void YTensor::SetDataLayout(DataLayout layout) {
    this->layout = layout;
}

TensorType YTensor::GetTensorType() {
    return this->tensor_type;
}

void YTensor::SetTensorType(TensorType type) {
    this->tensor_type = type;
}

int64_t YTensor::GetRank() {
    return this->rank;
}

int64_t YTensor::GetDataLen() {
    return this->data_len;
}

void YTensor::SetName(std::string name) {
    this->name = name;
}

std::string YTensor::GetName() {
    return this->name;
}

void YTensor::Print(int64_t len) {
    if (this->is_gpu) {
        checkCudaStatus(cudaMemcpy(this->cpu_ptr, this->gpu_ptr, this->sizeoftype * this->length,
                                   cudaMemcpyDeviceToHost));
    }
    printf("data: ");
    for (auto i=0; i<len; i++) {
        if(this->data_type == DataType::FLOAT32) {
            printf("%f, ", ((float*)this->cpu_ptr)[i]);
        }
    }
    printf("\n");
}

template <typename T>
void convert_nchw_to_nhwc_cpu(T* in, T* out, int batch, int channel, int height, int width) {
    int hw = height * width;
    int input_i = 0;
    int output_i = 0;
    for (auto b_i=0; b_i < batch; b_i++) {
        for (auto c_i=0; c_i < channel; c_i++) {
            for (auto hw_i=0; hw_i < hw; hw_i++) {
                input_i = b_i * channel * hw + c_i * hw + hw_i;
                output_i = b_i * channel * hw + hw_i * channel + c_i;
                out[output_i] = in[input_i];
            }
        }
    }
}

template <typename T>
void convert_nhwc_to_nchw_cpu(T* in, T* out, int batch, int channel, int height, int width) {
    int hw = height * width;
    int input_i = 0;
    int output_i = 0;
    for (auto b_i=0; b_i < batch; b_i++) {
        for (auto c_i=0; c_i < channel; c_i++) {
            for (auto hw_i=0; hw_i < hw; hw_i++) {
                output_i = b_i * channel * hw + c_i * hw + hw_i;
                input_i = b_i * hw * channel + hw_i * channel + c_i;
                out[output_i] = in[input_i];
            }
        }
    }
}


bool YTensor::ConvertLayout(DataLayout layout) {
    if (this->layout == layout) {
        return true;
    } else if ((!this->is_gpu) and this->layout == DataLayout::NCHW and layout == DataLayout::NHWC) {
        if (this->data_type == DataType::FLOAT32) {
            float* tmp = (float*)malloc(this->length * sizeof(float));
            convert_nchw_to_nhwc_cpu<float>((float*)this->cpu_ptr, tmp, this->shape.at(0), this->shape.at(1), this->shape.at(2), this->shape.at(3));
            free(this->cpu_ptr);
            this->cpu_ptr = tmp;
        } else if (this->data_type == DataType::FLOAT16) {
            half* tmp = (half*)malloc(this->length * sizeof(half));
            convert_nchw_to_nhwc_cpu<half>((half*)this->cpu_ptr, tmp, this->shape.at(0), this->shape.at(1), this->shape.at(2), this->shape.at(3));
            free(this->cpu_ptr);
            this->cpu_ptr = tmp;
        }
        this->data = this->cpu_ptr;
        this->layout = DataLayout::NHWC;
    } else if ((!this->is_gpu) and this->layout == DataLayout::NHWC and layout == DataLayout::NCHW) {
        if (this->data_type == DataType::FLOAT32) {
            float* tmp = (float*)malloc(this->length * sizeof(float));
            convert_nhwc_to_nchw_cpu<float>((float*)this->cpu_ptr, tmp, this->shape.at(0), this->shape.at(1), this->shape.at(2), this->shape.at(3));
            free(this->cpu_ptr);
            this->cpu_ptr = tmp;
        } else if (this->data_type == DataType::FLOAT16) {
            half* tmp = (half*)malloc(this->length * sizeof(half));
            convert_nhwc_to_nchw_cpu<half>((half*)this->cpu_ptr, tmp, this->shape.at(0), this->shape.at(1), this->shape.at(2), this->shape.at(3));
            free(this->cpu_ptr);
            this->cpu_ptr = tmp;
        }
        this->data = this->cpu_ptr;
        this->layout = DataLayout::NCHW;
    } else {
        printf("tensor ConvertLayout not support yet !!");
        return false;
    }
    return true;
}
