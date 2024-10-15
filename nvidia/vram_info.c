#include "./../library.h"

#include <cuda_runtime_api.h>


/** Get total VRAM size
 * @return total RAM in bytes
 */
UINT64 nvidiaTotalVram(void) {
    size_t freeMemory, totalMemory;
    cudaMemGetInfo(&freeMemory, &totalMemory);
    return totalMemory;
}


/** Get free RAM size
 * @return free RAM in bytes
 */
UINT64 nvidiaFreeVram(void) {
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
