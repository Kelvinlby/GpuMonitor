#include "library.h"

#include <stdio.h>
#include <sys/sysinfo.h>


ULONG getTotalRam(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    return info.totalram * info.mem_unit;
}


ULONG getFreeRam(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    return info.freeram * info.mem_unit;
}


UINT8 getRamUsage(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    return (UINT8) (100(1 - (double) (info.freeram * info.mem_unit) / (double) (info.totalram * info.mem_unit)));
}
