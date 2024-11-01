#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lidsor_filter.hpp"

namespace py = pybind11;

PYBIND11_MODULE(lidsor_filter, m) {
    m.doc() = "LIDSOR filter implementation in C++"; 
    
    m.def("filtering_lidsor_cpp", &filtering_lidsor_cpp,
          py::arg("points"),
          py::arg("k") = 30,
          py::arg("s") = 2.0f,
          py::arg("i_threshold") = 25.0f,
          py::arg("d_threshold") = 25.0f,
          py::arg("scaling_factor") = 110.0f,
          "Fast LIDSOR filtering implementation");
}