cmake_minimum_required(VERSION 3.29)
project(GpuMonitor C)

# Set custom cmake module path here
set(MY_CMAKE_MODULE_PATH "/home/kelvin/Application/JetBrains/clion/bin/cmake/linux/x64/share/cmake-3.29/Modules")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${MY_CMAKE_MODULE_PATH})

set(CMAKE_C_STANDARD 23)

add_library(GpuMonitor SHARED library.c
        nvidia/vram_info.c
        nvidia/processor_info.c
        amd/rocm_smi.h
        amd/processor_info.c
        amd/vram_info.c
)

#Change "/home/usr" to GpuMonitor installation directory
#IMPORTANT - It's strongly recommended that you should not work with "rocm_smi.h" from your local ROCm lib!
include_directories(/usr/local/cuda/include
        /home/usr/GpuMonitor/amd/rocm_smi.h
)

find_package(CUDA REQUIRED)
target_link_libraries(GpuMonitor ${CUDA_LIBRARIES} nvidia-ml)

# Change ../rocm-6.1.3/.. to your version
target_link_libraries(GpuMonitor /opt/rocm-6.1.3/lib/librocm_smi64.so)

