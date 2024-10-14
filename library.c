#include "library.h"

#include <stdio.h>
#include <sys/sysinfo.h>

/************************************ message to ECHO HELLO WORLD ************************************
 *
 * Since this is a shared library, u need to create a separate C project to test its functionality.
 * Don't put test code in this repo, and it's preferred to use Dart to test it, but C is also OK.
 * And, Dart ffi interface only support functions that returns the following types:
 *      - Int / Int8 / Int16 / Int32 / Int64
 *      - Long / LongLong
 *      - Uint / Uint8 / Uint16 / Uint32 / Uint64
 *
 * So if a function that is designed to be called by `Submission` project, it should return values
 * in the types above.
 *
 * By the way, remove this message if u have read it.   :)
 */


__u_long getTotalRam(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    return info.totalram * info.mem_unit;
}


__u_long getFreeRam(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    return info.freeram * info.mem_unit;
}


__uint8_t getRamUsage(void) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        return 1;
    }

    // return 1 - (double) (info.freeram * info.mem_unit) / (double) (info.totalram * info.mem_unit);
    // TODO scale the return.
    return 0;
}
