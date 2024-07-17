#include "../../include/kernels.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "../utils.h"
#include <cstdio>

template<typename T>
__global__ void elementwise_fp_cuda(T *in0, T *in1, T *out, int length) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length) {
        ((T*)out)[index] = ((T*)in0)[index] + ((T*)in1)[index];
        //printf("%d = %f \n", index, ((float*)out)[index]);
    }
}

bool elementwise(int64_t in_ptr0, int64_t in_ptr1, int64_t out_ptr,
                std::vector<int> in_shape0, std::vector<int> in_shape1,
                std::vector<int> out_shape, std::string dtype, std::string layout) {

    int block_size = 512;
    int length = 1;
    for (auto& shape : out_shape) {
        length *= shape;
    }

    int grid_size = (length + block_size - 1) / block_size;
    if (dtype == "float32") {
        elementwise_fp_cuda<float><<<grid_size, block_size>>>((float*)in_ptr0, (float*)in_ptr1, (float*)out_ptr,
                                            (int)length);
    } else if (dtype == "float16"){
        elementwise_fp_cuda<__half><<<grid_size, block_size>>>((__half*)in_ptr0, (__half*)in_ptr1, (__half*)out_ptr,
                                            (int)length);
    }
    
    return true;
}
