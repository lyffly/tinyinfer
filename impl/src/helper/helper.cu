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

struct GpuTimer
{
    cudaStream_t _stream_id;
    cudaEvent_t _start;
    cudaEvent_t _stop;

    /// Constructor
    GpuTimer() : _stream_id(0)
    {
        CUDA_CHECK(cudaEventCreate(&_start));
        CUDA_CHECK(cudaEventCreate(&_stop));
    }

    /// Destructor
    ~GpuTimer()
    {
        CUDA_CHECK(cudaEventDestroy(_start));
        CUDA_CHECK(cudaEventDestroy(_stop));
    }

    /// Start the timer for a given stream (defaults to the default stream)
    void start(cudaStream_t stream_id = 0)
    {
        _stream_id = stream_id;
        CUDA_CHECK(cudaEventRecord(_start, _stream_id));
    }

    /// Stop the timer
    void stop()
    {
        CUDA_CHECK(cudaEventRecord(_stop, _stream_id));
    }

    /// Return the elapsed time (in milliseconds)
    float elapsed_millis()
    {
        float elapsed = 0.0;
        CUDA_CHECK(cudaEventSynchronize(_stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
        return elapsed;
    }
};

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
    if (!this->is_gpu) {
        printf("[Error] data not on gpu !!\n");
        return false;
    }
    if (this->dtype == DataType::HALF) {
        this->dtype = DataType::HALF;
        this->sizeoftype = sizeof(float);
        int block_size = 512;
        int grid_size = (this->length + block_size - 1) / block_size;
        void* tmp;
        cudaMalloc((void**)&tmp, this->length * sizeof(half));
        cudaMemcpy(tmp, this->gpu_ptr, this->length * sizeof(half), cudaMemcpyDeviceToDevice);
        cudaMalloc((void**)&(this->gpu_ptr), this->length * this->sizeoftype);
        convert_fp16_to_fp32_cuda<<<grid_size, block_size>>>((half*)tmp, (float*)this->gpu_ptr,
                                                             this->length);
        this->data = this->gpu_ptr;
        cudaFree(tmp);
    } else if (this->dtype == DataType::FLOAT32) {
        return true;
    } else {
        printf("[Error] datatype not correct !!\n");
        return false;
    }
    return true;
}

bool YTensor::Half() {
    if (!this->is_gpu) {
        printf("[Error] data not on gpu !!\n");
        return false;
    }
    if (this->dtype == DataType::FLOAT32) {
        this->dtype = DataType::FLOAT32;
        this->sizeoftype = sizeof(half);
        int block_size = 512;
        int grid_size = (this->length + block_size - 1) / block_size;
        void* tmp;
        cudaMalloc((void**)&tmp, this->length * sizeof(float));
        cudaMemcpy(tmp, this->gpu_ptr, this->length * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMalloc((void**)&(this->gpu_ptr), this->length * this->sizeoftype);
        convert_fp32_to_fp16_cuda<<<grid_size, block_size>>>((float*)tmp, (half*)this->gpu_ptr,
                                                             this->length);
        this->data = this->gpu_ptr;
        cudaFree(tmp);
    } else if (this->dtype == DataType::HALF) {
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
