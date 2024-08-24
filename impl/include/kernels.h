#pragma once
#include <cudnn_v9.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "math.h"
#include "stdio.h"

using namespace std;

enum class DataType {
    UNDEFINE = 0,
    INT8 = 1,
    HALF = 2,
    FLOAT16 = 2,
    FLOAT32 = 3,
    INT32 = 4,
    INT64 = 5,
    BOOL = 6,
};

enum class DataLayout {
    UNDEFINE = 0,
    NCHW = 1,
    NHWC = 2,
};

enum class TensorType {
    UNDEFINE = 0,
    CONSTANT = 1,
    VARIABLE = 2,
    INPUT = 3,
    OUTPUT = 4,
};

class YTensor {
   public:
    int64_t rank;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    void* data;
    bool is_gpu;
    TensorType tensor_type;
    DataType data_type;
    DataLayout layout;
    int64_t length;
    int64_t sizeoftype;
    float scale;
    float* channel_scale;

   public:
    YTensor();
    ~YTensor();
    bool Malloc(std::vector<int64_t> shape, DataType dtype, DataLayout layout);
    bool Free();
    bool Zeros(std::vector<int64_t> shape, DataType dtype, DataLayout layout);
    bool Float();
    bool Half();
    bool CUDA();
    bool CPU();
    bool CopyNumpyData(int64_t ptr);
    int64_t GetDataPtr();
    void SetDataPtr(int64_t ptr, bool is_gpu);
    std::vector<int64_t> GetShape();
    void SetShape(std::vector<int64_t> shape);
    bool GetIsGPU();
    void SetIsGPU(bool is_gpu);
    DataType GetDataType();
    void SetDataType(DataType type);
    DataLayout GetDataLayout();
    void SetDataLayout(DataLayout layout);
    TensorType GetTensorType();
    void SetTensorType(TensorType type);
    int64_t GetRank();
    void SetRank(int64_t rank);



   private:
    void* cpu_ptr;
    void* gpu_ptr;
};

//***********************************************************************************************************
// Handle
void* create_handle();

//***********************************************************************************************************
// activation
bool activation_backend(int64_t in_ptr, int64_t out_ptr, float alpha, float beta,
                        std::vector<int> in_shape, std::vector<int> out_shape, std::string dtype,
                        std::string layout, std::string optype, int64_t pstream);

//***********************************************************************************************************
// elementwise
bool elementwise_backend(int64_t in_ptr0, int64_t in_ptr1, int64_t out_ptr,
                         std::vector<int> in_shape0, std::vector<int> in_shape1,
                         std::vector<int> out_shape, std::string dtype, std::string layout,
                         std::string optype, int64_t pstream);

//***********************************************************************************************************
// gemm
int64_t get_gemm_workspace_size();

bool gemm_cublas_backend(int64_t in_ptr, int64_t weight_ptr, int64_t bias_ptr, int64_t out_ptr,
                         int64_t workspace_size, int64_t workspace_ptr, float alpha, float beta,
                         bool transA, bool transB, std::vector<int> in_shape,
                         std::vector<int> weight_shape, std::vector<int> bias_shape,
                         std::vector<int> out_shape, std::string dtype, int64_t pstream);

bool gemv_cuda_backend(int64_t in_ptr, int64_t weight_ptr, int64_t bias_ptr, int64_t out_ptr,
                         int64_t workspace_size, int64_t workspace_ptr, float alpha, float beta,
                         bool transA, bool transB, std::vector<int> in_shape,
                         std::vector<int> weight_shape, std::vector<int> bias_shape,
                         std::vector<int> out_shape, std::string dtype, int64_t pstream);

bool gemm_cutlass_backend(int64_t in_ptr, int64_t weight_ptr, int64_t bias_ptr, int64_t out_ptr,
                          int64_t workspace_size, int64_t workspace_ptr, float alpha, float beta,
                          bool transA, bool transB, std::vector<int> in_shape,
                          std::vector<int> weight_shape, std::vector<int> bias_shape,
                          std::vector<int> out_shape, std::string dtype, int64_t pstream);

//***********************************************************************************************************
// data type convert
bool datatype_convert_backend(int64_t in_ptr, int64_t out_ptr, std::vector<int> in_shape,
                              std::vector<int> out_shape, std::string layout, std::string in_dtype,
                              std::string out_dtype, int64_t pstream);


//***********************************************************************************************************
// conv2d
void* create_conv2d_desc();

int64_t get_conv2d_workspace_size(std::vector<int> kernels, std::vector<int> paddings,
                                  std::vector<int> strides, std::vector<int> dilations, int group,
                                  std::vector<int> in_shape, std::vector<int> weight_shape,
                                  std::vector<int> bias_shape, std::vector<int> out_shape,
                                  std::string dtype, std::string layout, int64_t algo,
                                  int64_t pstream, void* desc);

bool conv2d_backend(int64_t in_ptr, int64_t weight_ptr, int64_t bias_ptr, int64_t out_ptr,
                    int64_t workspace_size, int64_t workspace_ptr, int64_t algo,
                    std::vector<int> kernels, std::vector<int> paddings, std::vector<int> strides,
                    std::vector<int> dilations, int group, std::vector<int> in_shape,
                    std::vector<int> weight_shape, std::vector<int> bias_shape,
                    std::vector<int> out_shape, std::string dtype, std::string layout,
                    int64_t pstream, void* desc);

int64_t get_conv2d_algo(std::vector<int> kernels, std::vector<int> paddings,
                        std::vector<int> strides, std::vector<int> dilations, int group,
                        std::vector<int> in_shape, std::vector<int> weight_shape,
                        std::vector<int> bias_shape, std::vector<int> out_shape, std::string dtype,
                        std::string layout, int64_t pstream, void* desc);

// int64_t get_conv2d_workspace_size(std::vector<int> kernels, std::vector<int> paddings,
//                                   std::vector<int> strides, std::vector<int> dilations, int group,
//                                   std::vector<int> in_shape, std::vector<int> weight_shape,
//                                   std::vector<int> bias_shape, std::vector<int> out_shape,
//                                   std::string dtype, std::string layout, int64_t algo,
//                                   int64_t pstream, void* desc);

// bool conv2d_cudnn_backend(int64_t in_ptr, int64_t weight_ptr, int64_t bias_ptr, int64_t out_ptr,
//                     int64_t workspace_ptr, std::string dtype, std::string layout,
//                     int64_t pstream, void* desc);

//***********************************************************************************************************
// pooling
void* create_pooling_desc();

void setup_pooling_descriptor(std::vector<int>& kernels, std::vector<int>& paddings,
                              std::vector<int>& strides, std::vector<int>& in_shape,
                              std::vector<int>& out_shape, std::string optype, std::string dtype,
                              std::string layout, void* desc);

bool pooling_cudnn_backend(int64_t in_ptr, int64_t out_ptr, std::vector<int> kernels,
                     std::vector<int> paddings, std::vector<int> strides, std::vector<int> in_shape,
                     std::vector<int> out_shape, std::string optype, std::string dtype,
                     std::string layout, int64_t pstream, void* desc);

bool pooling_cuda_backend(int64_t in_ptr, int64_t out_ptr, std::vector<int> kernels,
                     std::vector<int> paddings, std::vector<int> strides, std::vector<int> in_shape,
                     std::vector<int> out_shape, std::string optype, std::string dtype,
                     std::string layout, int64_t pstream, void* desc);

//***********************************************************************************************************
// convert data layout
bool layout_convert_backend(int64_t in_ptr, int64_t out_ptr, std::vector<int> in_shape,
                            std::vector<int> out_shape, std::string dtype, std::string in_layout,
                            std::string out_layout, int64_t pstream);

//***********************************************************************************************************
// gelu
bool gelu_cuda_backend(int64_t in_ptr, int64_t out_ptr, std::vector<int> in_shape, std::vector<int> out_shape,
        std::string dtype, std::string layout, int64_t pstream);

//***********************************************************************************************************
// rms norm
bool rms_norm_cuda_backend(int64_t in_ptr, int64_t out_ptr, std::vector<int> in_shape,
                         std::vector<int> out_shape, float eps, std::string dtype, int64_t pstream);


//***********************************************************************************************************
// silu
bool silu_cuda_backend(int64_t in_ptr, int64_t out_ptr, std::vector<int> in_shape, std::vector<int> out_shape,
        std::string dtype, std::string layout, int64_t pstream);

