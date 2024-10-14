#include "library.h"

#include <stdio.h>
#include <sys/sysinfo.h>


/** Get total RAM size
 * @return total RAM in bytes
 */
ULONG getTotalRam(void) {
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
ULONG getFreeRam(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    return info.freeram * info.mem_unit;
}


/** Get system RAM utilization
 * @return 1000 times the utilization rate
 */
UINT16 getRamUsage(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    return (UINT16) (1000 * (1 - (FLOAT32) (info.freeram * info.mem_unit) / (FLOAT32) (info.totalram * info.mem_unit)));
}
