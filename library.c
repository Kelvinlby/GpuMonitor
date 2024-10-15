#include "library.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/sysinfo.h>

#include <rocm_smi.h>

/** Get which platform is supported (Nvidia platform is prioritized)
 * @return platform code:   0 - CPU only;    1 - Nvidia GPU available;  2 - AMD GPU available;
 */
UINT8 getPlatform(void) {
    // Nvidia platform checking
    int deviceCount = 0;
    const cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id == cudaSuccess && deviceCount > 0) {
        return 1;
    }

    // AMD platform checking
    rsmi_init(0);
    UINT64 devid;
    rsmi_status_t ret = rsmi_dev_guid_get(0, &devid);

    if (ret == RSMI_STATUS_SUCCESS) {
      return 2
    }
    rsmi_shut_down();

    return 0;
}

/*
 * |---------------------------SYSTEM FUNCTIONS---------------------------|
 */

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

/*
 * |------------------------------AMD FUNCTIONS------------------------------|
 */

/** Get total vRAM size
 * @return total RAM in bytes
 */
UINT64 amdTotalVram(void) {
    rsmi_init(0);

    UINT64 total = 0;
    rsmi_status_t ret = rsmi_dev_memory_total_get(0, RSMI_MEM_TYPE_VRAM, &total);

    if (ret != RSMI_STATUS_SUCCESS) {
        printf("Failed to get total memory");
        return 1;
    }

    return total;

    rsmi_shut_down();
}

/** Get free vRAM size
 * @return free RAM in bytes
 */
UINT64 amdFreeVram(void) {
    rsmi_init(0);

    UINT64 used = 0, free = 0;
    UINT64 total = A_getTotalRam();
    rsmi_status_t ret = rsmi_dev_memory_usage_get(0, RSMI_MEM_TYPE_VRAM, &used);

    if (ret != RSMI_STATUS_SUCCESS) {
        printf("Failed to get free memory");
        return 1;
        }

    free = used - total;
    return free;

    rsmi_shut_down();
}

/** Get system vRAM utilization
 * @return percentage of utilization rate
 */
UINT8 amdVramUsage(void) {
    rsmi_init(0);

    UINT8 percentage = 0;
    UINT64 used = 0, total = A_getTotalRam();
    rsmi_status_t ret = rsmi_dev_memory_usage_get(0, RSMI_MEM_TYPE_VRAM, &used);

    if (ret != RSMI_STATUS_SUCCESS) {
        printf("Failed to get usage memory");
        return 1;
    }

    percentage = used / total;
    return percentage;

    rsmi_shut_down();
}