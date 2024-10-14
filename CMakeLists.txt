cmake_minimum_required(VERSION 3.29)
project(GpuMonitor C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake /home/kelvin/Application/JetBrains/clion/bin/cmake/linux/x64/share/cmake-3.29/Modules)

set(CMAKE_C_STANDARD 23)

add_library(GpuMonitor SHARED library.c
        nvidia/vram_info.c
        nvidia/cuda_info.c
        nvidia/tensor_info.c)

include_directories(/usr/local/cuda/include)
find_package(CUDA REQUIRED)
target_link_libraries(GpuMonitor ${CUDA_LIBRARIES})
