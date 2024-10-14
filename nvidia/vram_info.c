#include "./../library.h"

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvml.h>


/** Get total VRAM size
 * @return total RAM in bytes
 */
ULONG nvidiaTotalVram(void) {
    size_t freeMemory, totalMemory;
    cudaMemGetInfo(&freeMemory, &totalMemory);
    return totalMemory;
}


/** Get free RAM size
 * @return free RAM in bytes
 */
ULONG nvidiaFreeVram(void) {
    size_t freeMemory, totalMemory;
    cudaMemGetInfo(&freeMemory, &totalMemory);
    return freeMemory;
}


/** Get system RAM utilization
 * @return 1000 times the utilization rate
 */
UINT16 nvidiaVramUsage(void) {
    size_t freeMemory, totalMemory;
    cudaMemGetInfo(&freeMemory, &totalMemory);
    return (UINT16) (1000 * (1 - (FLOAT32) freeMemory / (FLOAT32) totalMemory));
}
