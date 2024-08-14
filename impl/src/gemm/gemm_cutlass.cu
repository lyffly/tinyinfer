#include <complex>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include "../helper/helper.h"

#include "../../include/kernels.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "cutlass/gemm/device/gemm.h"

using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

using CutlassGemmFloat = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                     RowMajor,     // Layout of A matrix
                                                     float,        // Data-type of B matrix
                                                     RowMajor,     // Layout of B matrix
                                                     float,        // Data-type of C matrix
                                                     RowMajor>;    // Layout of C matrix

using CutlassGemmHalf = cutlass::gemm::device::Gemm<__half,       // Data-type of A matrix
                                                    RowMajor,     // Layout of A matrix
                                                    __half,       // Data-type of B matrix
                                                    RowMajor,     // Layout of B matrix
                                                    __half,       // Data-type of C matrix
                                                    RowMajor>;    // Layout of C matrix


bool gemm_cutlass_backend(int64_t in_ptr, int64_t weight_ptr, int64_t bias_ptr, int64_t out_ptr,
                          int64_t workspace_size, int64_t workspace_ptr, float alpha, float beta,
                          bool transA, bool transB, std::vector<int> in_shape,
                          std::vector<int> weight_shape, std::vector<int> bias_shape,
                          std::vector<int> out_shape, std::string dtype, int64_t pstream) {
    cudaStream_t stream = (cudaStream_t)pstream;

    int m = in_shape.at(0);
    int k = in_shape.at(1);
    int n = 0;
    // 1*512 @ 512*1000 = 1*1000
    // 1*512 @ 1000*512 = 1*1000 transb
    int lda = k;    // m * k
    int ldb = n;    // k * n
    int ldc = n;    // m * n

    if (transB) {    // n*k
        n = weight_shape.at(0);
        ldb = k;
        ldc = n;
    } else {    // k*n
        n = weight_shape.at(1);
        ldb = n;
        ldc = n;
    }

    printf("m %d n %d k %d , lda %d ldb %d ldc %d \n\n", m, n, k, lda, ldb, ldc);
    printf("ptr %p  %p %p  \n\n", (float*)in_ptr, (float*)weight_ptr, (float*)out_ptr);

    printf("alpha %f \n", alpha);

    if (dtype == "float32") {
        printf(" 111  \n");
        using Gemm = cutlass::gemm::device::Gemm<float,                           // ElementA
                                                 cutlass::layout::RowMajor,       // LayoutA
                                                 float,                           // ElementB
                                                 cutlass::layout::ColumnMajor,    // LayoutB
                                                 float,                           // ElementOutput
                                                 cutlass::layout::RowMajor        // LayoutOutput
                                                 >;
        Gemm fp32_gemm;
        float beta0 = 0.0f;
        printf(" 122  \n");
        Gemm::Arguments args({m, n, k},                    // problem size
                             {(float*)in_ptr, lda},        // A
                             {(float*)weight_ptr, ldb},    // B
                             {(float*)out_ptr, ldc},       // C
                             {(float*)out_ptr, ldc},       // D
                             {alpha, beta0});
        printf(" 123  \n");
        CUTLASS_CHECK(fp32_gemm(args));
        printf(" 124  \n");
    } else if (dtype == "float16") {
        // CutlassGemmHalf fp16_gemm;
        // CutlassGemmHalf::Arguments args({m, n, k},                     // problem size
        //                                 {(__half*)in_ptr, lda},        // A
        //                                 {(__half*)weight_ptr, ldb},    // B
        //                                 {(__half*)out_ptr, ldc},       // C
        //                                 {(__half*)out_ptr, ldc},       // D
        //                                 {alpha, beta});
        // fp16_gemm(args);
    } else {
        printf("[Error] Gemm not support data type %s !!! \n", dtype.c_str());
    }
}