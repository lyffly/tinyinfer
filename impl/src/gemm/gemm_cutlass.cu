#include <iostream>
#include <sstream>
#include <vector>

#include "cuda_runtime.h"
#include "cutlass/gemm/device/gemm.h"

using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

using CutlassGemm = cutlass::gemm::device::Gemm<float,           // Data-type of A matrix
                                                ColumnMajor,     // Layout of A matrix
                                                float,           // Data-type of B matrix
                                                ColumnMajor,     // Layout of B matrix
                                                float,           // Data-type of C matrix
                                                ColumnMajor>;    // Layout of C matrix