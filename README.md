# GpuMonitor
Access detailed GPU information with this shared library.

## Supported platform
- AMD Radeon RX 7900 XTX
- AMD Radeon RX 7900 XT
- AMD Radeon RX 7900 GRE
- AMD Radeon PRO W7900
- AMD Radeon PRO W7900DS
- AMD Radeon PRO W7800
> For full AMD compatibility information, refer to [ROCm](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
> Notice: code for different platforms are located in different branches

## Pre-requirements
- Linux system required (tested on Ubuntu)
- ROCm 6.1.3 (or newer)

## Build Guide
- Install ROCm
- Set permissions for groups
  - To check your group in your system, use this command:
    ```shell
    groups
    ```
  - Add yourself to 'render' and 'video' group:
    ```shell
    sudo usermod -a -G render,video $LOGNAME
    ```
- Disable integrated graphics (IGP) to avoid crash
