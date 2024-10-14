# GpuMonitor
Access detailed GPU information with this shared library.

## Supported platform
- Nvidia Series
  - Nvidia Datacenter GPU
  - Nvidia RTX Professional GPU
  - Nvidia GeForce GPU
- AMD Series
  - AMD Radeon RX 7900 XTX
  - AMD Radeon RX 7900 XT
  - AMD Radeon RX 7900 GRE
  - AMD Radeon PRO W7900
  - AMD Radeon PRO W7900DS
  - AMD Radeon PRO W7800

> For full AMD Compatibility matrix, refer to [ROCm Doc](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)

## Pre-requirements
- Linux system required (tested on Ubuntu)
- CUDA 12.6 (or newer) and ROCm 6.1.3 (or newer)

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
