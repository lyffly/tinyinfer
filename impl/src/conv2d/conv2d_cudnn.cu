#include "../../include/kernels.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cublasLt.h"
#include "cublas_v2.h"
#include "cudnn.h"
#include "stdio.h"
#include "math.h"
#include <iostream>

#include <cstdint>
#include <vector>
#include <string>
#include <cstdio>
#include "../helper/helper.h"


bool conv2d_backend(int64_t in_ptr, int64_t weight_ptr, int64_t bias_ptr, int64_t out_ptr, int64_t workspace_size, 
                int64_t workspace_ptr, float alpha, float beta, bool transA, bool transB, 
                std::vector<int> in_shape, std::vector<int> weight_shape, std::vector<int> bias_shape,
                std::vector<int> out_shape, std::string dtype, int64_t pstream) {



    return true;
}