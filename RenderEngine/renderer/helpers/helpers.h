/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "config.h"

// Printf issues
//rtPrintf()
//		- doesn't print spaces in a loop
//		- consecutive calls sometimes cause format exception - http://celarek.at/2014/05/why-you-should-never-use-nvidia-optix/
// printf():
//		- not officially supported, sometimes causes crashes
//		- doesn't print spaces in a loop based on depth value (SOMETIMES ?) if depth variable used in the print call before loop
//      - sometimes fails printing values of size_t2.y and size_t3.y correctly
// The issues does complicate debugging process, hence the multiple switches for the macro below for quick switching


#define OPTIX_DEBUG_STD_PRINTF 1
#define OPTIX_PRINTFI_IDX 1         // printing multiple consecutive spaces seems random - doesn't always work
#define OPTIX_DEBUG_ID_X 0
#define OPTIX_DEBUG_ID_Y 0


#if OPTIX_DEBUG_STD_PRINTF || !defined(__CUDACC__)
#include <stdio.h>
#define OPTIX_PRINTF_FUN printf
#else
#define OPTIX_PRINTF_FUN rtPrintf
#endif

#if ENABLE_RENDER_DEBUG_OUTPUT

// OPTIX_XXX_DISABLE may disable printf in some individual file to avoid recompilation of everything
#ifndef OPTIX_PRINTFI_DISABLE
#define OPTIX_PRINTFI(depth, str, ...) \
    if (launchIndex.x == OPTIX_DEBUG_ID_X && launchIndex.y == OPTIX_DEBUG_ID_Y) \
    {  \
        if (OPTIX_PRINTFI_IDX) \
        { \
            OPTIX_PRINTF_FUN("%d, %d - d %d - ", launchIndex.x, launchIndex.y, depth); \
        } \
        OPTIX_PRINTF_FUN(str, __VA_ARGS__); \
    }
#else
#define OPTIX_PRINTFI(depth, str, ...)
#endif

#ifndef OPTIX_PRINTFID_DISABLE
#define OPTIX_PRINTFID(launchId, str, ...) \
    if (launchId.x == OPTIX_DEBUG_ID_X && launchId.y == OPTIX_DEBUG_ID_Y) \
    {  \
        if (OPTIX_PRINTFI_IDX) \
        { \
            OPTIX_PRINTF_FUN("%d, %d - d X - ", launchId.x, launchId.y); \
        } \
        OPTIX_PRINTF_FUN(str, __VA_ARGS__); \
    }
#else
#define OPTIX_PRINTFID(depth, str, ...)
#endif

#ifndef OPTIX_PRINTFIALL_DISABLE
#define OPTIX_PRINTFIALL(depth, str, ...) \
    if (OPTIX_PRINTFI_IDX) \
    { \
        OPTIX_PRINTF_FUN("%d, %d - d %d - ", launchIndex.x, launchIndex.y, depth); \
    } \
    OPTIX_PRINTF_FUN(str, __VA_ARGS__);
#else
#define OPTIX_PRINTFIALL(depth, str, ...) 
#endif

// Added explicit macros for rtPrintf since printf fails to print fields of size_t2, size_t3 correctly
#ifndef OPTIX_RTPRINTFI_DISABLE
#define OPTIX_RTPRINTFI(depth, str, ...) \
    if (launchIndex.x == OPTIX_DEBUG_ID_X && launchIndex.y == OPTIX_DEBUG_ID_Y) \
    {  \
        if (OPTIX_PRINTFI_IDX) \
        { \
            rtPrintf("%d, %d - d %d - ", launchIndex.x, launchIndex.y, depth); \
        } \
        rtPrintf(str, __VA_ARGS__); \
    }
#else
#define OPTIX_RTPRINTFI(depth, str, ...) 
#endif

#ifndef OPTIX_RTPRINTFID_DISABLE
#define OPTIX_RTPRINTFID(depth, str, ...) \
    if (launchId.x == OPTIX_DEBUG_ID_X && launchId.y == OPTIX_DEBUG_ID_Y) \
    {  \
        if (OPTIX_PRINTFI_IDX) \
        { \
            rtPrintf("%d, %d - d X - ", launchId.x, launchId.y); \
        } \
        rtPrintf(str, __VA_ARGS__); \
    }
#else
#define OPTIX_RTPRINTFID(depth, str, ...) 
#endif

#ifdef __CUDACC__
#ifndef OPTIX_PRINTF_DISABLE
#define OPTIX_PRINTF(str, ...) OPTIX_PRINTF_FUN(str, __VA_ARGS__);
#endif
#else
#define OPTIX_PRINTF(str, ...)
#endif


#else // !ENABLE_RENDER_DEBUG_OUTPUT

#define OPTIX_PRINTF(str, ...)
#define OPTIX_PRINTFI(depth, str, ...)
#define OPTIX_PRINTFID(depth, str, ...)
#define OPTIX_PRINTFIALL(depth, str, ...)
#define OPTIX_RTPRINTFI(depth, str, ...)
#define OPTIX_RTPRINTFID(depth, str, ...)

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