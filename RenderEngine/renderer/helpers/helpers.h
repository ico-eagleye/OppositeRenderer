/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "config.h"

// Optix inlines all functions, Cuda compiles sometimes fails to inline many parameter functions with __inline__ hint
// so this is precaution macro
#define RT_FUNCTION __forceinline__ __device__

// Printf issues
//rtPrintf()
//		- doesn't print spaces in a loop
//		- consecutive calls sometimes cause format exception - http://celarek.at/2014/05/why-you-should-never-use-nvidia-optix/
//      - many rtPrintf calls can cause slow context compilation (maybe due many launchIndex conditions?)
// printf():
//		- not officially supported, sometimes causes crashes, weird behavior (buffer indices corrupt and out of bounds etc)
//		- doesn't print spaces in a loop based on depth value (SOMETIMES ?) if depth variable used in the print call before loop
//      - sometimes fails printing values of size_t2.y and size_t3.y correctly
// The issues does complicate debugging process, hence the multiple switches for the macro below for quick switching


#define OPTIX_DEBUG_STD_PRINTF 0
#define OPTIX_PRINTFI_IDX 1         // printing multiple consecutive spaces seems random - doesn't always work
#define OPTIX_DEBUG_ID_X 0
#define OPTIX_DEBUG_ID_Y 0


#if OPTIX_DEBUG_STD_PRINTF || !defined(__CUDACC__)
#include <stdio.h>
#define OPTIX_PRINTF_FUN printf
#else
#define OPTIX_PRINTF_FUN rtPrintf
#endif

// OPTIX_XX_DEF     allow to enable disable given printf macro in the file
// OPTIX_XXX_ENABLE may disable printf in some individual part of the file
#if ENABLE_RENDER_DEBUG_OUTPUT

#ifdef OPTIX_PRINTFID_DEF
#define OPTIX_PRINTFID(launchIdx, depth, str, ...) \
    if (OPTIX_PRINTFID_ENABLED && launchIdx.x == OPTIX_DEBUG_ID_X && launchIdx.y == OPTIX_DEBUG_ID_Y) \
    {  \
        OPTIX_PRINTF_FUN("%u, %u - d %u - " str, launchIdx.x, launchIdx.y, depth, __VA_ARGS__); \
    }
#else
#define OPTIX_PRINTFID(depth, str, ...) 
#endif

#ifdef OPTIX_PRINTFI_DEF
#define OPTIX_PRINTFI(launchIdx, str, ...) \
    if (OPTIX_PRINTFI_ENABLED && launchIdx.x == OPTIX_DEBUG_ID_X && launchIdx.y == OPTIX_DEBUG_ID_Y) \
    {  \
        OPTIX_PRINTF_FUN("%u, %u - d X - " str, launchIdx.x, launchIdx.y, __VA_ARGS__); \
    }
#else
#define OPTIX_PRINTFI(str, launchIdx, ...) 
#endif

#ifdef OPTIX_PRINTF_DEF
#define OPTIX_PRINTF(str, ...) \
if (OPTIX_PRINTF_ENABLED) \
    OPTIX_PRINTF_FUN(str, __VA_ARGS__);
#else
#define OPTIX_PRINTF(depth, str, ...) 
#endif

#else // !ENABLE_RENDER_DEBUG_OUTPUT

#define OPTIX_PRINTF(str, ...)
#define OPTIX_PRINTFI(depth, str, ...)
#define OPTIX_PRINTFID(depth, str, ...)

#endif




// Create ONB from normalized normal (code: Physically Based Rendering, Pharr & Humphreys pg. 63)
static  __device__ __inline__ void createCoordinateSystem( const optix::float3& N, optix::float3& U, optix::float3& V/*, optix::float3& W*/ )
{
    using namespace optix;

    if(fabs(N.x) > fabs(N.y))
    {
        float invLength = 1.f/sqrtf(N.x*N.x + N.z*N.z);
        U = make_float3(-N.z*invLength, 0.f, N.x*invLength);
    }
    else
    {
        float invLength = 1.f/sqrtf(N.y*N.y + N.z*N.z);
        U = make_float3(0.f, N.z*invLength, -N.y*invLength);
    }
    V = cross(N, U);
}

static __device__ __host__ __forceinline__ float maxf(float a, float b)
{
    return a > b ? a : b;
}

// Returns true if ray direction points in the opposite direction 
// as the normal, where the normal points outwards from the face
static __device__ __host__ __inline__ bool hitFromOutside(const optix::float3 & rayDirection, const optix::float3 & normal)
{
    return (optix::dot(normal, rayDirection) < 0);
}

static __device__ __forceinline__ int intmin(int a, int b)
{
    return a < b ? a : b;
}

static __device__ __forceinline__ int intmin(unsigned int a, unsigned int b)
{
    return a < b ? a : b;
}

static __device__ __forceinline__ float favgf(const optix::float3 & v )
{
    return (v.x+v.y+v.z)*0.3333333333f;
}

template<typename T>
__device__ __forceinline__ T sqr(const T& a) { return a*a; }


static __device__ __forceinline__ bool isZero(const optix::float3 & v )
{
    return v.x == 0.f && v.y == 0.f && v.z == 0.f;
}


__host__ __device__ __inline__ unsigned int getBufIndex1D(
    const optix::uint3 & index3D, const optix::uint3& bufSize )
{
    return index3D.x + index3D.y * bufSize.x + index3D.z * bufSize.x * bufSize.y;
}

__host__ __device__ __inline__ unsigned int getBufIndex1D(
    const optix::uint2 & index2D, const optix::uint2& bufSize )
{
    return index2D.x + index2D.y * bufSize.x;
}

RT_FUNCTION bool isNAN(float f)
{
    return f != f;
}

RT_FUNCTION bool isNAN(optix::float2 f)
{
    return f.x != f.x || f.y != f.y;
}

RT_FUNCTION bool isNAN(optix::float3 f)
{
    return f.x != f.x || f.y != f.y || f.z != f.z;
}