/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "config.h"
//#define PRINTF printf
#define PRINTF rtPrintf // doesn't show up when output forwarded to a file

#if ENABLE_RENDER_DEBUG_OUTPUT

#if PRINTF
#define OPTIX_DEBUG_PRINT(depth, str, ...) \
    if (launchIndex.x == 0 && launchIndex.y == 0) \
    {  \
    PRINTF("%d %d: ", launchIndex.x, launchIndex.y); \
    for(int i = 0; i < depth; i++) { PRINTF(" "); } \
    PRINTF(str, __VA_ARGS__); \
    }

#else

// With rtPrintf use single output. Multiple consecutive can "Error in rtPrintf format string"
// exception - http://celarek.at/2014/05/why-you-should-never-use-nvidia-optix/
// Some say it's indication of memory corruption somewhere.. investigating.
#define OPTIX_DEBUG_PRINT(depth, str, ...) \
    if (launchIndex.x == 0 && launchIndex.y == 0) \
    {  \
    PRINTF(str, __VA_ARGS__); \
    }
#endif
//#define OPTIX_DEBUG_PRINT(depth, str, ...) \
//    if (launchIndex.x == 0 && launchIndex.y == 0) \
//    {  \
//        PRINTF("i %d, %d - d %d - ", launchIndex.x, launchIndex.y, depth); \
//        for(int i = 0; i < depth; i++) { PRINTF(" "); } \
//        PRINTF(str, __VA_ARGS__); \
//    }
//#endif

#else
#define OPTIX_DEBUG_PRINT(depth, str, ...) // nothing
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

static __device__ __host__ __inline__ float maxf(float a, float b)
{
    return a > b ? a : b;
}

// Returns true if ray direction points in the opposite direction 
// as the normal, where the normal points outwards from the face
static __device__ __host__ __inline__ bool hitFromOutside(const optix::float3 & rayDirection, const optix::float3 & normal)
{
    return (optix::dot(normal, rayDirection) < 0);
}

static __device__ __inline__ int intmin(int a, int b)
{
    return a < b ? a : b;
}

static __device__ __inline__ float favgf(const optix::float3 & v )
{
    return (v.x+v.y+v.z)*0.3333333333f;
}