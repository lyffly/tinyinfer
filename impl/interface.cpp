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
    m.def("elementwise", &elementwise);
    m.def("activation", &activation);

}
