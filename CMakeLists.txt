cmake_minimum_required(VERSION 3.29)
project(GpuMonitor C)

# Set custom cmake module path here
set(MY_CMAKE_MODULE_PATH "/home/kelvin/Application/JetBrains/clion/bin/cmake/linux/x64/share/cmake-3.29/Modules" "home/patrick/clion-2024.2.1/bin/cmake/linux/x64/share/cmake-3.29/Modules")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${MY_CMAKE_MODULE_PATH})

set(CMAKE_C_STANDARD 23)

add_library(GpuMonitor SHARED library.c
        nvidia/vram_info.c
        nvidia/processor_info.c
        amd/rocm_smi.h
        amd/processor_info.c
        amd/vram_info.c
)

include_directories(/usr/local/cuda/include)
include_directories(/opt/rocm/include)

find_package(CUDA REQUIRED)
target_link_libraries(GpuMonitor ${CUDA_LIBRARIES} nvidia-ml)

find_package(ROCM REQUIRED)
target_link_libraries(GpuMonitor ${ROCM_LIBRARIES})



