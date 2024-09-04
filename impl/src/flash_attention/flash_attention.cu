#include <iostream>
#include "../../include/kernels.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "math.h"
#include "stdio.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>


bool flash_attention_backend(int64_t in_ptr0, int64_t in_ptr1, int64_t out_ptr,
                             std::vector<int> in_shape0, std::vector<int> in_shape1,
                             std::vector<int> out_shape, std::string dtype, int64_t pstream) {
    cudaStream_t stream = (cudaStream_t)pstream;
    int block_size = 512;
    int length = 1;
    for (auto& shape : out_shape) {
        length *= shape;
    }

    return true;
}
