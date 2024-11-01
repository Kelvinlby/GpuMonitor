#include "./../library.h"
#include <rocm_smi.h>


/** Get percentage of time device is busy doing any processing
 * @return percentage of core utilization
 */
UINT32 amdGpuUsage(void){
    rsmi_init(0);

    UINT32 usage;
    rsmi_dev_busy_percent_get(0, &usage);

    UINT32 usage_buf = usage;
    rsmi_shut_down();

    return usage_buf;
}
