#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lidsor_filter.hpp"

namespace py = pybind11;

PYBIND11_MODULE(lidsor_filter, m) {
    m.doc() = "LIDSOR filter implementation in C++"; 
    
    m.def("filtering_lidsor_cpp", &filtering_lidsor_cpp,
          py::arg("points"),
          py::arg("k_neighbors") = 30,
          py::arg("std_multiplier") = 2.0f,
          py::arg("range_multiplier") = 1.0f,
          py::arg("intensity_threshold") = 25.0f,
          py::arg("distance_threshold") = 25.0f,
          py::arg("scaling_factor") = 110.0f,
          "Fast LIDSOR filtering implementation");
}