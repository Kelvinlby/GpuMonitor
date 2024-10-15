#ifndef GPUMONITOR_LIBRARY_H
#define GPUMONITOR_LIBRARY_H

#include <bits/types.h>

/********** Data Types  *********/
// Unsigned integers
#define UINT8  __uint8_t
#define UINT16 __uint16_t
#define UINT32 __uint32_t
#define UINT64 __uint64_t

// Signed integers
#define INT8  __int8_t
#define INT16 __int16_t
#define INT32 __int32_t
#define INT64 __int64_t

// Floating points
#define FLOAT32 float
#define FLOAT64 double

/****  Function Declarations  ****/
// CPU
UINT8  getPlatform(void);
UINT64 getTotalRam(void);
UINT64 getFreeRam(void);
UINT8  getRamUsage(void);

// Nvidia
UINT64 nvidiaTotalVram(void);
UINT64 nvidiaFreeVram(void);
UINT8  nvidiaVramUsage(void);
UINT8  nvidiaCudaUsage(void);

// AMD

#endif //GPUMONITOR_LIBRARY_H
