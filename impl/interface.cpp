#include "include/kernels.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "stdio.h"
#include "math.h"
#include <iostream>
#include <cstdint>
#include <vector>
#include <string>


namespace py = pybind11;


PYBIND11_MODULE(kernels, m) {
    m.def("elementwise", &elementwise_backend);
    m.def("activation", &activation_backend);
    m.def("gemm", &gemm_backend);
    m.def("cast", &cast_backend);
    m.def("conv2d", &conv2d_backend);
    m.def("get_conv2d_algo", &get_conv2d_algo);
    m.def("get_conv2d_workspace_size", &get_conv2d_workspace_size);
    

    py::enum_<DataType>(m, "DataType")
        .value("INT8", DataType::INT8)
        .value("HALF", DataType::HALF)
        .value("FLOAT16", DataType::FLOAT16)
        .value("FLOAT32", DataType::FLOAT32)
        .value("INT32", DataType::INT32)
        .value("INT64", DataType::INT64)
        .value("BOOL", DataType::BOOL)
        .export_values();

    py::enum_<DataLayout>(m, "DataLayout")
        .value("NCHW", DataLayout::NCHW)
        .value("NHWC", DataLayout::NHWC)
        .export_values();

    py::class_<Dims>(m, "Dims")
        .def(py::init())
        .def_readwrite("nbDims", &Dims::nbDims)
        .def_readwrite("d", &Dims::d);


    py::class_<YTensor>(m, "YTensor")
        .def(py::init())
        .def_property ("data_ptr", &YTensor::GetDataPtr, &YTensor::SetDataPtr)
        .def_property("shape", &YTensor::GetShape, &YTensor::SetShape);

}
