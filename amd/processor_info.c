#include "./../library.h"
#include "./rocm_smi.h"


/** Get utilization of amd gpu
 * @return percentage of Cuda core utilization
 */
UINT32 amdGpuUsage(void){
    rsmi_init(0)

    UINT32 useage;
    rsmi_dev_busy_percent_get(0, &useage);

    UINT32 useage_buf = useage;
    rsmi_shut_down();

    return useage;
}