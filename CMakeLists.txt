cmake_minimum_required(VERSION 3.29)
project(GpuMonitor C)

set(CMAKE_C_STANDARD 23)

add_library(GpuMonitor SHARED library.c
        nvidia/vram_info.c
        nvidia/cuda_info.c
        nvidia/tensor_info.c)

include_directories(/usr/local/cuda/include)
