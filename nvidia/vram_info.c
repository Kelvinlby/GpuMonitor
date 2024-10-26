#include "./../library.h"

#include <cuda_runtime_api.h>


/** Get total VRAM size
 * @return total VRAM in MB
 */
UINT32 nvidiaTotalVram(void) {
    size_t freeMemory, totalMemory;
    cudaMemGetInfo(&freeMemory, &totalMemory);
    return totalMemory / (1024 * 1024);
}


/** Get used VRAM size
 * @return used VRAM in MB
 */
UINT32 nvidiaUsedVram(void) {
    size_t freeMemory, totalMemory;
    cudaMemGetInfo(&freeMemory, &totalMemory);
    return (totalMemory - freeMemory) / (1024 * 1024);
}


/** Get system VRAM utilization
 * @return percent of utilization rate
 */
UINT8 nvidiaVramUsage(void) {
    size_t freeMemory, totalMemory;
    cudaMemGetInfo(&freeMemory, &totalMemory);
    return (UINT8) (100 * (1 - (FLOAT32) freeMemory / (FLOAT32) totalMemory));
}
