#pragma once
#include "stdio.h"
#include "math.h"
#include <iostream>
#include <cstdint>
#include <vector>
#include <string>


class Dims {
public:
    int nbDims;
    int d[8];
};

enum class DataType {
    INT8=1,
    HALF=2,
    FLOAT16=2,
    FLOAT32=3,
    INT32=4,
    INT64=5,
    BOOL=6,
};

enum class DataLayout {
    NCHW = 1,
    NHWC = 2,
};

class YTensor {
public:
    Dims dims;
    void* data;
    bool is_gpu;
    DataType type;
    DataLayout layout;
    
private:
    void* cpu_ptr;
    void* gpu_ptr;

};


bool activation(int64_t in_ptr, int64_t out_ptr, float alpha, float beta,
                std::vector<int> in_shape, std::vector<int> out_shape, std::string dtype, 
                std::string layout, std::string optype);


bool elementwise(int64_t in_ptr0, int64_t in_ptr1, int64_t out_ptr,
                std::vector<int> in_shape0, std::vector<int> in_shape1,
                std::vector<int> out_shape, std::string dtype, std::string layout, std::string optype);







