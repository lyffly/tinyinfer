#pragma once
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "math.h"
#include "stdio.h"

class Dims {
   public:
    int nbDims;
    std::vector<int> d;
};

enum class DataType {
    INT8 = 1,
    HALF = 2,
    FLOAT16 = 2,
    FLOAT32 = 3,
    INT32 = 4,
    INT64 = 5,
    BOOL = 6,
};

enum class DataLayout {
    NCHW = 1,
    NHWC = 2,
};

class YTensor {
   public:
    int nbDims;
    std::vector<int> d;
    void* data;
    bool is_gpu;
    DataType type;
    DataLayout layout;
    float scale;
    float* channel_scale;

   public:
    YTensor();
    ~YTensor();
    bool Malloc(Dims dims, DataType dtype, DataLayout layout, int64_t length);
    bool Free();
    bool Zeros(Dims dims, DataType dtype);
    bool Zeros(Dims dims, DataType dtype, DataLayout layout);
    bool Float();
    bool Half();
    bool CUDA();
    bool CPU();
    int64_t GetDataPtr();
    void SetDataPtr(int64_t ptr);
    Dims GetShape();
    void SetShape(Dims dims);

   private:
    void* cpu_ptr;
    void* gpu_ptr;
};

bool activation_backend(int64_t in_ptr, int64_t out_ptr, float alpha, float beta,
                        std::vector<int> in_shape, std::vector<int> out_shape, std::string dtype,
                        std::string layout, std::string optype, int64_t pstream);

bool elementwise_backend(int64_t in_ptr0, int64_t in_ptr1, int64_t out_ptr,
                         std::vector<int> in_shape0, std::vector<int> in_shape1,
                         std::vector<int> out_shape, std::string dtype, std::string layout,
                         std::string optype, int64_t pstream);

int64_t get_gemm_workspace_size();

bool gemm_backend(int64_t in_ptr, int64_t weight_ptr, int64_t bias_ptr, int64_t out_ptr,
                  int64_t workspace_size, int64_t workspace_ptr, float alpha, float beta,
                  bool transA, bool transB, std::vector<int> in_shape,
                  std::vector<int> weight_shape, std::vector<int> bias_shape,
                  std::vector<int> out_shape, std::string dtype, int64_t pstream);

bool cast_backend(int64_t in_ptr, int64_t out_ptr, std::vector<int> in_shape,
                  std::vector<int> out_shape, std::string layout, std::string in_dtype,
                  std::string out_dtype, int64_t pstream);

//***********************************************************************************************************
// conv2d
int64_t get_conv2d_algo(std::vector<int> kernels, std::vector<int> paddings,
                        std::vector<int> strides, std::vector<int> dilations, int group,
                        std::vector<int> in_shape, std::vector<int> weight_shape,
                        std::vector<int> bias_shape, std::vector<int> out_shape, std::string dtype,
                        std::string layout);

int64_t get_conv2d_workspace_size(std::vector<int> kernels, std::vector<int> paddings,
                                  std::vector<int> strides, std::vector<int> dilations, int group,
                                  std::vector<int> in_shape, std::vector<int> weight_shape,
                                  std::vector<int> bias_shape, std::vector<int> out_shape,
                                  std::string dtype, std::string layout, int64_t algo);

bool conv2d_backend(int64_t in_ptr, int64_t weight_ptr, int64_t bias_ptr, int64_t out_ptr,
                    int64_t workspace_size, int64_t workspace_ptr, int64_t algo,
                    std::vector<int> kernels, std::vector<int> paddings, std::vector<int> strides,
                    std::vector<int> dilations, int group, std::vector<int> in_shape,
                    std::vector<int> weight_shape, std::vector<int> bias_shape,
                    std::vector<int> out_shape, std::string dtype, std::string layout,
                    int64_t pstream);
