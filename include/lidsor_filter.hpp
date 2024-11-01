#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <nanoflann.hpp>

namespace py = pybind11;

std::vector<std::array<float, 4>> filtering_lidsor_cpp(
    const py::array_t<float>& points,
    int k = 30,
    float s = 2.0f,
    float i_threshold = 25.0f,
    float d_threshold = 25.0f,
    float scaling_factor = 110.0f
);