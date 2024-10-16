#include "./../library.h"
#include "./rocm_smi.h"


/** Get total VRAM size
 * @return total VRAM in bytes
 */
UINT64 amdTotalVram(void) {
    rsmi_init(0);

    UINT64 total = 0;
    rsmi_dev_memory_total_get(0, RSMI_MEM_TYPE_VRAM, &total);

    UINT64 total_buf = total;
    rsmi_shut_down();

    return total_buf;
}

/** Get used VRAM size
 * @return used RAM in bytes
 */
UINT64 amdUsedVram(void) {
    rsmi_init(0);

    UINT64 used = 0;
    rsmi_dev_memory_usage_get(0, RSMI_MEM_TYPE_VRAM, &used);
    UINT64 used_buf = used;
    rsmi_shut_down();

    return used_buf;
}

/** Get system VRAM utilization
 * @return percentage of utilization rate
 */
UINT8 amdVramUsage(void) {
    rsmi_init(0);

    UINT8 percentage = 0;
    UINT64 used = 0, total = 0;
    rsmi_dev_memory_total_get(0, RSMI_MEM_TYPE_VRAM, &total);
    rsmi_dev_memory_usage_get(0, RSMI_MEM_TYPE_VRAM, &used);

    percentage = (UINT8) (100 * (FLOAT32) used / (FLOAT32) total);
    rsmi_shut_down();

    return percentage;
}
