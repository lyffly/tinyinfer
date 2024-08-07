#include <cstddef>
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

int GetSizeofDtype(DataType dtype) {
    if (dtype == DataType::INT32 || dtype == DataType::FLOAT32) {
        return 4;
    } else if (dtype == DataType::FLOAT16 || dtype == DataType::HALF) {
        return 2;
    } else if (dtype == DataType::INT64) {
        return 8;
    } else if (dtype == DataType::INT8 || dtype == DataType::BOOL) {
        return 1;
    }
    return 1;
}

size_t GetProdofVector(std::vector<int> shapes) {
    size_t sum = 1;
    for (auto& shape : shapes) {
        sum *= shape;
    }
    return sum;
}

YTensor::YTensor() {}

YTensor::~YTensor() {
    Free();
}

bool YTensor::Malloc(Dims dims, DataType dtype, DataLayout layout) {
    this->sizeoftype = GetSizeofDtype(dtype);
    this->length = GetProdofVector(dims.shapes);
    this->cpu_ptr = malloc(this->sizeoftype * this->length);
    this->data = cpu_ptr;
    this->gpu_ptr = nullptr;
    this->nb_dims = dims.nb_dims;
    this->shape = dims.shapes;
    this->dtype = dtype;
    this->layout = layout;
    this->is_gpu = false;
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

bool YTensor::Zeros(Dims dims, DataType dtype, DataLayout layout) {
    this->sizeoftype = GetSizeofDtype(dtype);
    this->length = GetProdofVector(dims.shapes);
    this->cpu_ptr = malloc(this->sizeoftype * this->length);
    if (not this->cpu_ptr) {
        std::cout << "malloc error \n";
    }
    memset(this->cpu_ptr, 0, this->sizeoftype * this->length);
    this->data = cpu_ptr;
    this->gpu_ptr = nullptr;
    this->nb_dims = dims.nb_dims;
    this->shape = dims.shapes;
    this->dtype = dtype;
    this->layout = layout;
    this->is_gpu = false;
    return true;
}

bool YTensor::Float() {
    return true;
}

bool YTensor::Half() {
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
    return true;
}

void YTensor::SetDataPtr(int64_t ptr) {
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

void YTensor::SetShape(std::vector<int> dims) {
    this->shape = dims;
}

std::vector<int> YTensor::GetShape() {
    return this->shape;
}


void* create_handle() {
    Handles* handle = new Handles();
    return (void*)handle;
}
