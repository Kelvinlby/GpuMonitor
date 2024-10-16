#include "./../library.h"
#include "./rocm_smi.h"


/** Get total vRAM size
 * @return total RAM in bytes
 */
UINT64 amdTotalVram(void) {
    rsmi_init(0);

    UINT64 total = 0;
    rsmi_status_t ret = rsmi_dev_memory_total_get(0, RSMI_MEM_TYPE_VRAM, &total);

    if (ret != RSMI_STATUS_SUCCESS) {
        return 1;
    }

    UINT64 total_buf = total;
    rsmi_shut_down();

    return total_buf;
}

/** Get free vRAM size
 * @return free RAM in bytes
 */
UINT64 amdFreeVram(void) {
    rsmi_init(0);

    UINT64 used = 0, free = 0, total = 0;
    rsmi_status_t ret1 = rsmi_dev_memory_total_get(0, RSMI_MEM_TYPE_VRAM, &total);
    rsmi_status_t ret2 = rsmi_dev_memory_usage_get(0, RSMI_MEM_TYPE_VRAM, &used);

    if (ret1 != RSMI_STATUS_SUCCESS || ret2 != RSMI_STATUS_SUCCESS) {
        return 1;
    }

    free = used - total;
    rsmi_shut_down();

    return free;
}

/** Get system vRAM utilization
 * @return percentage of utilization rate
 */
UINT8 amdVramUsage(void) {
    rsmi_init(0);

    UINT8 percentage = 0;
    UINT64 used = 0, total = 0;
    rsmi_status_t ret1 = rsmi_dev_memory_total_get(0, RSMI_MEM_TYPE_VRAM, &total);
    rsmi_status_t ret2 = rsmi_dev_memory_usage_get(0, RSMI_MEM_TYPE_VRAM, &used);

    if (ret1 != RSMI_STATUS_SUCCESS || ret2 != RSMI_STATUS_SUCCESS) {
        return 1;
    }

    percentage = (UINT8) (100 * (FLOAT32) used / (FLOAT32) total);
    rsmi_shut_down();

    return percentage;
}
