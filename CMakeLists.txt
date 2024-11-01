cmake_minimum_required(VERSION 3.10)
project(lidsor_filter)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# pybind11の設定
find_package(pybind11 REQUIRED)

# nanoflannのインクルードパス設定
include_directories(${CMAKE_SOURCE_DIR}/third_party/nanoflann/include)

# モジュールの作成
pybind11_add_module(lidsor_filter 
    src/lidsor_filter.cpp
    src/lidsor_module.cpp
)

target_include_directories(lidsor_filter PRIVATE include)