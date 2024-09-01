#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include "../helper/helper.h"

#include "../../include/kernels.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"


bool silu_cuda_backend(int64_t in_ptr, int64_t out_ptr, std::vector<int> in_shape, std::vector<int> out_shape,
        std::string dtype, std::string layout, int64_t pstream) {


    return true;
}
