#include "library.h"

#include <stdio.h>
#include <sys/sysinfo.h>

unsigned long getTotalRam(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    return info.totalram * info.mem_unit;
}

unsigned long getFreeRam(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    return info.freeram * info.mem_unit;
}

double getRamUsage(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    return 1 - (double) (info.freeram * info.mem_unit) / (double) (info.totalram * info.mem_unit);
}
