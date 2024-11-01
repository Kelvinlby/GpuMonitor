#include "library.h"

#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
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
    UINT64 deviceId;
    const rsmi_status_t status = rsmi_dev_guid_get(0, &deviceId);

    if (status == RSMI_STATUS_SUCCESS) {
        return 2;
    }

    rsmi_shut_down();

    return 0;
}


/** Get total RAM size
 * @return total RAM in MB
 */
UINT32 getTotalRam(void) {
    UINT32 total = 0;

    FILE *f = fopen("/proc/meminfo", "r");

    if (f == NULL) {
        perror("Error opening /proc/meminfo");
    }

    char line[64];

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemTotal:", 9) == 0) {
            sscanf(line + 9, "%u", &total);
            break;
        }
    }

    fclose(f);

    return total / 1024;
}


/** Get free RAM size
 * @return free RAM in MB
 */
UINT32 getFreeRam(void) {
    UINT32 available = 0;

    FILE *f = fopen("/proc/meminfo", "r");

    if (f == NULL) {
        perror("Error opening /proc/meminfo");
    }

    char line[64];

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemAvailable:", 13) == 0) {
            sscanf(line + 13, "%u", &available);
            break;
        }
    }

    fclose(f);

    return available / 1024;
}


/** Get RAM usage
 * @return Ram usage in percentage
 */
UINT8 getRamUsage(void) {
    UINT32 available = 0;
    UINT32 total = 0;
    bool available_ready = false;
    bool total_ready = false;

    FILE *f = fopen("/proc/meminfo", "r");

    if (f == NULL) {
        perror("Error opening /proc/meminfo");
    }

    char line[64];

    while (fgets(line, sizeof(line), f)) {
        if (!available_ready && strncmp(line, "MemAvailable:", 13) == 0) {
            sscanf(line + 13, "%u", &available);
            available_ready = true;
        }

        if (!total_ready && strncmp(line, "MemTotal:", 9) == 0) {
            sscanf(line + 9, "%u", &total);
            total_ready = true;
        }

        if(total_ready && available_ready) {
            break;
        }
    }

    fclose(f);

    return 100 * (1 - (FLOAT32) available / (FLOAT32) total);
}
