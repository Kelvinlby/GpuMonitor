#include "library.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/sysinfo.h>


/** Get which platform is supported (Nvidia platform is prioritized)
 * @return platform code:   0 - CPU only;    1 - Nvidia GPU available;  2 - AMD GPU available;
 */
UINT8 getPlatform(void) {
    int deviceCount;
    const cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        return 1;
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
        printf("Result: CUDA is NOT available\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
        printf("Result: CUDA is available\n");
    }

    return 0;
}


/** Get total RAM size
 * @return total RAM in bytes
 */
UINT64 getTotalRam(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    return info.totalram * info.mem_unit;
}


/** Get free RAM size
 * @return free RAM in bytes
 */
UINT64 getFreeRam(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    return info.freeram * info.mem_unit;
}


/** Get system RAM utilization
 * @return percentage of utilization rate
 */
UINT8 getRamUsage(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    return (UINT8) (100 * (1 - (FLOAT32) (info.freeram * info.mem_unit) / (FLOAT32) (info.totalram * info.mem_unit)));
}
