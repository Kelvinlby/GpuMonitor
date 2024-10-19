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
UINT32 getTotalRam(void);
UINT32 getFreeRam(void);
UINT8  getRamUsage(void);

// Nvidia
UINT32 nvidiaTotalVram(void);
UINT32 nvidiaUsedVram(void);
UINT8  nvidiaVramUsage(void);
UINT8  nvidiaGpuUsage(void);

// AMD
UINT32 amdTotalVram(void);
UINT32 amdUsedVram(void);
UINT8  amdVramUsage(void);
UINT32 amdGpuUsage(void);

#endif //GPUMONITOR_LIBRARY_H
