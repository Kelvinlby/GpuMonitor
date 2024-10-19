#include "library.h"

#include <stdio.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <cuda_runtime.h>
#include "./amd/rocm_smi.h"


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
    UINT64 deviceId;
    rsmi_status_t status = rsmi_dev_guid_get(0, &deviceId);

    if (status == RSMI_STATUS_SUCCESS) {
      return 2;
    }
    rsmi_shut_down();

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


/** Get cached memory size
 * @return cached RAM in bytes
 */
UINT64 getCached() {
    FILE *f = fopen("/proc/meminfo", "r");
    if (f == NULL) {
        perror("Error opening /proc/meminfo");
    }
    char line[256];
    UINT64 cached = 0;

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "Cached:", 7) == 0) {
            sscanf(line + 7, "%ld", &cached);
            break;
        }
    }
    fclose(f);
    return cached * 1024;
}


/** Get available memory size
 * @return available RAM in bytes
 */
UINT64 getAvailable(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    FILE *f = fopen("/proc/meminfo", "r");
    if (f == NULL) {
        perror("Error opening /proc/meminfo")
        return 2;
    }

    char line[256];
    UINT64 cached = 0;

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "Cached:", 7) == 0) {
            sscanf(line + 7, "%ld", &cached);
            break;
        }
    }
    fclose(f);

    UINT64 free = info.freeram * info.mem_unit;
    UINT64 available = free + cached * 1024;
    return available;
}