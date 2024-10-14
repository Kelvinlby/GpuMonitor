#include "./../library.h"

#include <cuda.h>
#include <nvml.h>

/** Get total VRAM size
 * @return total RAM in bytes
 */
ULONG nvidiaTotalVram(void) {
    return 0;
}


/** Get free RAM size
 * @return free RAM in bytes
 */
ULONG nvidiaFreeVram(void) {
    return 0;
}


/** Get system RAM utilization
 * @return 1000 times the utilization rate
 */
UINT16 nvidiaVramUsage(void) {
    return 0;
}
