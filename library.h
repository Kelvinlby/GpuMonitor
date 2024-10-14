#ifndef GPUMONITOR_LIBRARY_H
#define GPUMONITOR_LIBRARY_H

#include <bits/types.h>

/********** Data Types  *********/
// unsigned integers
#define UINT8 __uint8_t
#define UINT16 __uint16_t
#define UINT32 __uint32_t
#define UINT64 __uint64_t
#define ULONG __u_long

// signed integers
#define INT8 __int8_t
#define INT16 __int16_t
#define INT32 __int32_t
#define INT64 __int64_t

// floating points
#define FLOAT32 float
#define FLOAT64 double

/****  Function Declarations  ****/
ULONG getTotalRam(void);
ULONG getFreeRam(void);
UINT16 getRamUsage(void);

#endif //GPUMONITOR_LIBRARY_H
