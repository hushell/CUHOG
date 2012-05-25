// 
// This file is derived from fastHOG, some utility functions are used in CUHOG
//


#ifndef __HOG_UTILS__
#define __HOG_UTILS__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include <cuda_gl_interop.h>
#include "cutil_inline.h"
#include <cuda.h>
#include "HOGDefines.h"

__host__ int iDivUp(int a, int b);
__host__ int iDivDown(int a, int b);
__host__ int iAlignUp(int a, int b);
__host__ int iAlignDown(int a, int b);

__host__ int iDivUpF(int a, float b);
__host__ int iClosestPowerOfTwo(int x);

__host__ void Float4ToUchar4(float4 *inputImage, uchar4 *outputImage, int width, int height);
__host__ void Float2ToUchar4(float2 *inputImage, uchar4 *outputImage, int width, int height, int index);
__host__ void Float2ToUchar1(float2 *inputImage, uchar1 *outputImage, int width, int height, int index);
__host__ void Float1ToUchar4(float1 *inputImage, uchar4 *outputImage, int width, int height);
__host__ void Float1ToUchar1(float1 *inputImage, uchar1 *outputImage, int width, int height);

__global__ void float4toUchar4(float4 *inputImage, uchar4 *outputImage, int width, int height);
__global__ void float2toUchar4(float2 *inputImage, uchar4 *outputImage, int width, int height, int index);
__global__ void float2toUchar1(float2 *inputImage, uchar1 *outputImage, int width, int height, int index);
__global__ void float1toUchar4(float1 *inputImage, uchar4 *outputImage, int width, int height);
__global__ void float1toUchar1(float1 *inputImage, uchar1 *outputImage, int width, int height);

__host__ void Uchar4ToFloat4(uchar4 *inputImage, float4 *outputImage, int width, int height);
__global__ void uchar4tofloat4(uchar4 *inputImage, float4 *outputImage, int width, int height);

#endif
