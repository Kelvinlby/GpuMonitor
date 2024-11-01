cmake_minimum_required(VERSION 3.29)
project(GpuMonitor C)

# Set custom cmake module path here
set(MY_CMAKE_MODULE_PATH "/home/patrick/JetBrains_IDE/Clion/clion-2024.2.1/bin/cmake/linux/x64/share/cmake-3.29/Modules")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${MY_CMAKE_MODULE_PATH})

set(CMAKE_C_STANDARD 23)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "/opt/rocm/lib:/opt/rocm/lib64")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_library(GpuMonitor SHARED library.c
        nvidia/vram_info.c
        nvidia/processor_info.c
        amd/processor_info.c
        amd/vram_info.c
)


link_directories(/opt/rocm/lib)
link_directories(/opt/rocm/lib64)


include_directories(/usr/local/cuda/include)
include_directories(/opt/rocm/share/rocm/cmake)
include_directories(/opt/rocm/include/rocm_smi)
include_directories(/opt/rocm/include)
include_directories(/opt/rocm/include/hsa)


find_package(ROCM REQUIRED)
find_library(ROCM_SMI_LIB
        NAMES rocm_smi64
        PATHS /opt/rocm/lib /opt/rocm/lib64
        REQUIRED
)
target_link_libraries(GpuMonitor ${ROCM_LIBRARIES} ${ROCM_SMI_LIB})


find_package(CUDA REQUIRED)
target_link_libraries(GpuMonitor ${CUDA_LIBRARIES} nvidia-ml)

