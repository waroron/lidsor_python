#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <nanoflann.hpp>

namespace py = pybind11;

std::tuple<std::vector<std::array<float, 4>>, std::vector<size_t>, std::vector<size_t>> filtering_lidsor_cpp(
    const py::array_t<float>& points,
    int k_neighbors = 30,
    float std_multiplier = 2.0f,
    float range_multiplier = 1.0f,
    float intensity_threshold = 25.0f,
    float distance_threshold = 25.0f,
    float scaling_factor = 110.0f
);