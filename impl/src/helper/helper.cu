#include "../../include/kernels.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "stdio.h"
#include "math.h"
#include <iostream>


YTensor::YTensor() {

}
YTensor::~YTensor() {
    
}
bool YTensor::Malloc(Dims dims, DataType dtype, DataLayout layout, int64_t length) {
    return true;
}
bool YTensor::Free() {
    return true;
}
bool YTensor::Zeros(Dims dims, DataType dtype) {
    return true;
}
bool YTensor::Zeros(Dims dims, DataType dtype, DataLayout layout) {
    return true;
}
bool YTensor::Float() {
    return true;
}
bool YTensor::Half() {
    return true;
}
bool YTensor::CPU() {
    
    return true;
}


void YTensor::SetDataPtr(int64_t ptr) {

}

int64_t YTensor::GetDataPtr() {
    return (int64_t)data;
}

void YTensor::SetShape(Dims dims) {

}

Dims YTensor::GetShape() {
    Dims dims;
    dims.nbDims = this->nbDims;
    for (auto i=0; i<dims.nbDims; i++) {
        dims.d[i] = this->d[i];
    }
    return dims;
}
