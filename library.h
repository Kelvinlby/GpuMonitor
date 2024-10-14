#ifndef GPUMONITOR_LIBRARY_H
#define GPUMONITOR_LIBRARY_H

#include <bits/types.h>

/********** Data Types  *********/
// Unsigned integers
#define UINT8 __uint8_t
#define UINT16 __uint16_t
#define UINT32 __uint32_t
#define UINT64 __uint64_t
#define ULONG __u_long

// Signed integers
#define INT8 __int8_t
#define INT16 __int16_t
#define INT32 __int32_t
#define INT64 __int64_t

// Floating points
#define FLOAT32 float
#define FLOAT64 double

/****  Function Declarations  ****/
// CPU
ULONG getTotalRam(void);
ULONG getFreeRam(void);
UINT16 getRamUsage(void);

// Nvidia
ULONG nvidiaTotalVram(void);
ULONG nvidiaFreeVram(void);
UINT16 nvidiaVramUsage(void);

// AMD

#endif //GPUMONITOR_LIBRARY_H
