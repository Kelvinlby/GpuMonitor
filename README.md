# GpuMonitor - `NVIDIA` Branch
Access detailed GPU information with this shared library.

## Supported platform
- Nvidia Datacenter GPU
- Nvidia RTX Professional GPU
- Nvidia GeForce GPU

## Requirements
- CUDA `12.6`

## Build Guide
This project uses `FindCUDA` module from Cmake. However, you are not able to use it by default according to policy CMP0146.
To enable the module, you need to remove the following lines in `FindCUDA.cmake`
```cmake
cmake_policy(GET CMP0146 _FindCUDA_CMP0146)
if(_FindCUDA_CMP0146 STREQUAL "NEW")
  message(FATAL_ERROR "The FindCUDA module has been removed by policy CMP0146.")
endif()

if(CMAKE_GENERATOR MATCHES "Visual Studio")
  cmake_policy(GET CMP0147 _FindCUDA_CMP0147)
  if(_FindCUDA_CMP0147 STREQUAL "NEW")
    message(FATAL_ERROR "The FindCUDA module does not work in Visual Studio with policy CMP0147.")
  endif()
endif()
```
