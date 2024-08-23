#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "include/kernels.h"
#include "math.h"
#include "stdio.h"

namespace py = pybind11;

PYBIND11_MODULE(kernels, m) {
    m.def("elementwise", &elementwise_backend);
    m.def("activation", &activation_backend);
    m.def("gemm", &gemm_cublas_backend);
    m.def("gemv", &gemv_cuda_backend);
    m.def("gemm_cutlass", &gemm_cutlass_backend);
    m.def("datatype_convert", &datatype_convert_backend);
    m.def("conv2d", &conv2d_backend);
    m.def("create_conv2d_desc", &create_conv2d_desc);
    m.def("get_conv2d_workspace_size", &get_conv2d_workspace_size);
    m.def("get_conv2d_algo", &get_conv2d_algo);
    m.def("layout_convert", &layout_convert_backend);
    m.def("create_handle", &create_handle);
    m.def("create_pooling_desc", &create_pooling_desc);
    m.def("setup_pooling_descriptor", &setup_pooling_descriptor);
    m.def("pooling", &pooling_cudnn_backend);
    m.def("gelu", &gelu_cuda_backend);
    m.def("silu", &silu_cuda_backend);
    m.def("rms_norm", &rms_norm_cuda_backend);


    py::enum_<DataType>(m, "DataType")
        .value("int8", DataType::INT8)
        .value("half", DataType::HALF)
        .value("float16", DataType::FLOAT16)
        .value("float32", DataType::FLOAT32)
        .value("int32", DataType::INT32)
        .value("int64", DataType::INT64)
        .value("bool", DataType::BOOL)
        .export_values();

    py::enum_<DataLayout>(m, "DataLayout")
        .value("nchw", DataLayout::NCHW)
        .value("nhwc", DataLayout::NHWC)
        .export_values();

    py::class_<Dims>(m, "Dims")
        .def(py::init())
        .def_readwrite("nb_dims", &Dims::nb_dims)
        .def_readwrite("shape", &Dims::shapes);

    py::class_<YTensor>(m, "YTensor")
        .def(py::init())
        .def_property("shape", &YTensor::GetShape, &YTensor::SetShape)
        .def("malloc", &YTensor::Malloc)
        .def("free", &YTensor::Free)
        .def("copy_numpy_data", &YTensor::CopyNumpyData)
        .def("zeros", &YTensor::Zeros)
        .def("data_ptr", &YTensor::GetDataPtr)
        .def("set_data_ptr", &YTensor::SetDataPtr)
        .def("float", &YTensor::Float)
        .def("half", &YTensor::Half)
        .def("cuda", &YTensor::CUDA)
        .def("cpu", &YTensor::CPU);
        
}
