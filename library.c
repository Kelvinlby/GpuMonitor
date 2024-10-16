#include "library.h"

#include <stdio.h>
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
