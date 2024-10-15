#include "./../library.h"

#include <cuda_runtime.h>
#include <nvml.h>


/** Get average Cuda core utilization of all Nvidia GPUs
 * @return percentage of Cuda core utilization
 */
UINT8 nvidiaGpuUsage(void) {
    INT32 deviceCount;
    FLOAT32 gpuUsage = 0.0f;
    nvmlDevice_t device;
    nvmlUtilization_t utilization;

    cudaGetDeviceCount(&deviceCount);

    nvmlInit_v2();

    for (INT32 i = 0; i < deviceCount; i++) {
        nvmlDeviceGetHandleByIndex(i, &device);
        nvmlDeviceGetUtilizationRates(device, &utilization);
        gpuUsage = gpuUsage + (FLOAT32) utilization.gpu / (FLOAT32) deviceCount;
    }

    nvmlShutdown();

    return (UINT8) gpuUsage;
}
