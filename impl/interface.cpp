#include "include/kernels.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(kernels, m) {
    m.def("elementwise", &elementwise);

}
