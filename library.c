#include "library.h"

#include <stdio.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <cuda_runtime.h>


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

    return 0;
}


/** Get total RAM size
 * @return total RAM in MB
 */
UINT32 getTotalRam(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    return info.totalram * info.mem_unit / (1024 * 1024);
}


/** Get free RAM size
 * @return free RAM in MB
 */
UINT32 getFreeRam(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    UINT32 cached = 0;

    FILE *f = fopen("/proc/meminfo", "r");

    if (f == NULL) {
        perror("Error opening /proc/meminfo");
    }

    char line[256];

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "Cached:", 7) == 0) {
            sscanf(line + 7, "%u", &cached);
            break;
        }
    }

    fclose(f);

    return cached / 1024 + info.freeram * info.mem_unit / (1024 * 1024);
}


/** Get RAM usage
 * @return Ram usage in percentage
 */
UINT8 getRamUsage(void) {
     struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    UINT32 cached = 0;

    FILE *f = fopen("/proc/meminfo", "r");
    if (f == NULL) {
        perror("Error opening /proc/meminfo");
    }
    char line[256];

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "Cached:", 7) == 0) {
            sscanf(line + 7, "%u", &cached);
            break;
        }
    }

    fclose(f);

    return (UINT8) (100 * (1 - (FLOAT32) (cached * 1024 + info.freeram * info.mem_unit) / (FLOAT32) (info.totalram * info.mem_unit)));
}