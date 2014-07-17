/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "config.h"
#include "renderer/device_common.h"

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
//#define OPTIX_DEBUG_ID_X 245 // light pt
//#define OPTIX_DEBUG_ID_Y 460
#define OPTIX_DEBUG_ID_X 50
#define OPTIX_DEBUG_ID_Y 18

#define OPTIX_DEBUG_PIX 0
#define OPTIX_DEBUG_PIX_X 40
#define OPTIX_DEBUG_PIX_Y 400

#define IS_DEBUG_ID(launchIdx) (launchIdx.x == OPTIX_DEBUG_ID_X && launchIdx.y == OPTIX_DEBUG_ID_Y)
#define IS_DEBUG_PIX(pixelIndex) (pixelIndex.x == OPTIX_DEBUG_PIX_X && pixelIndex.y == OPTIX_DEBUG_PIX_Y)

#if OPTIX_DEBUG_STD_PRINTF || !defined(__CUDACC__)
#include <stdio.h>
#define OPTIX_PRINTF_FUN printf
#else
#define OPTIX_PRINTF_FUN rtPrintf
#endif

// OPTIX_XXX_DEF    allow to enable disable given printf macro in the file
// OPTIX_XXX_ENABLE may disable printf in some individual part of the file
#if ENABLE_RENDER_DEBUG_OUTPUT

#if defined(OPTIX_PRINTFID_DEF)
// prints if debug launch id
#define OPTIX_PRINTFID(launchIdx, depth, str, ...) \
    if (OPTIX_PRINTFID_ENABLED && launchIdx.x == OPTIX_DEBUG_ID_X && launchIdx.y == OPTIX_DEBUG_ID_Y) \
    {  \
        OPTIX_PRINTF_FUN("%u, %u - d %u - " str, launchIdx.x, launchIdx.y, depth, __VA_ARGS__); \
    }
#else
// prints if debug launch id
#define OPTIX_PRINTFID(depth, str, ...) 
#endif

#if defined(OPTIX_PRINTFI_DEF)
// prints if debug launch id
#define OPTIX_PRINTFI(launchIdx, str, ...) \
    if (OPTIX_PRINTFI_ENABLED && launchIdx.x == OPTIX_DEBUG_ID_X && launchIdx.y == OPTIX_DEBUG_ID_Y) \
    {  \
        OPTIX_PRINTF_FUN("%u, %u - d X - " str, launchIdx.x, launchIdx.y, __VA_ARGS__); \
    }
#else
// prints if debug launch id
#define OPTIX_PRINTFI(str, launchIdx, ...) 
#endif

#if defined(OPTIX_PRINTF_DEF)
#define OPTIX_PRINTF(str, ...) \
if (OPTIX_PRINTF_ENABLED) \
    OPTIX_PRINTF_FUN(str, __VA_ARGS__);
#else
#define OPTIX_PRINTF(str, ...) 
#endif


#if defined(OPTIX_PRINTFC_DEF)
// prints if condition cond true
#define OPTIX_PRINTFC(cond, str, ...) \
    if (OPTIX_PRINTFC_ENABLED && (cond) ) \
        OPTIX_PRINTF_FUN(str, __VA_ARGS__);
#else
// prints if condition cond true
#define OPTIX_PRINTFC(cond, str, ...) 
#endif

#if defined(OPTIX_PRINTFCID_DEF)
// prints if condition cond true, prints launch index and depth in front
#define OPTIX_PRINTFCID(cond, launchIdx, depth, str, ...) \
    if (OPTIX_PRINTFCID_ENABLED && (cond) ) \
        OPTIX_PRINTF_FUN("%u, %u - d %u - " str, launchIdx.x, launchIdx.y, depth, __VA_ARGS__);
#else
// prints if condition cond true, prints launch index and depth in front
#define OPTIX_PRINTFCID(cond, str, ...) 
#endif

#else // !ENABLE_RENDER_DEBUG_OUTPUT

#define OPTIX_PRINTF(str, ...)
#define OPTIX_PRINTFI(launchIdx, str, ...)
#define OPTIX_PRINTFID(launchIdx, depth, str, ...)
#define OPTIX_PRINTFC(cond, str, ...) 
#define OPTIX_PRINTFCID(cond, launchIdx, depth, str, ...) 

#endif




// Create ONB from normalized normal (code: Physically Based Rendering, Pharr & Humphreys pg. 63)
static  RT_FUNCTION void createCoordinateSystem( const optix::float3& N, optix::float3& U, optix::float3& V/*, optix::float3& W*/ )
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

static __host__ RT_FUNCTION float maxf(float a, float b)
{
    return a > b ? a : b;
}

// Returns true if ray direction points in the opposite direction 
// as the normal, where the normal points outwards from the face
static __host__ RT_FUNCTION bool hitFromOutside(const optix::float3 & rayDirection, const optix::float3 & normal)
{
    return (optix::dot(normal, rayDirection) < 0);
}

static RT_FUNCTION int intmin(int a, int b)
{
    return a < b ? a : b;
}

static RT_FUNCTION int intmin(unsigned int a, unsigned int b)
{
    return a < b ? a : b;
}

static RT_FUNCTION float favgf(const optix::float3 & v )
{
    return (v.x+v.y+v.z)*0.3333333333f;
}

template<typename T>
RT_FUNCTION T sqr(const T& a) { return a*a; }


static RT_FUNCTION bool isZero(const optix::float3 & v )
{
    return v.x == 0.f && v.y == 0.f && v.z == 0.f;
}


__host__ RT_FUNCTION unsigned int getBufIndex1D(
    const optix::uint3 & index3D, const optix::uint3& bufSize )
{
    return index3D.x + index3D.y * bufSize.x + index3D.z * bufSize.x * bufSize.y;
}

__host__ RT_FUNCTION unsigned int getBufIndex1D(
    const optix::uint2 & index2D, const optix::uint2& bufSize )
{
    return index2D.x + index2D.y * bufSize.x;
}


template<typename T>
RT_FUNCTION bool _isNaN(T v)
{
    return v != v;
}

RT_FUNCTION bool isNaN(float v)
{
    return _isNaN(v);
}

RT_FUNCTION bool isNaN(optix::float2 v)
{
    return _isNaN(v.x) || _isNaN(v.y);
}

RT_FUNCTION bool isNaN(optix::float3 v)
{
    return _isNaN(v.x) || _isNaN(v.y) || _isNaN(v.z);
}

#if defined(__CUDACC__)
#define CUDART_INF_F_POS __int_as_float(0x7f800000)
#define CUDART_INF_F_NEG __int_as_float(0xff800000)
#else
#define CUDART_INF_F_POS std::numeric_limits<float>::infinity() 
#define CUDART_INF_F_NEG -1*std::numeric_limits<float>::infinity()
#endif

RT_FUNCTION bool isInf(float v)
{
    return v == CUDART_INF_F_POS || v == CUDART_INF_F_NEG;
}

RT_FUNCTION bool isInf(optix::float2 v)
{
    return isInf(v.x) || isInf(v.y);
}

RT_FUNCTION bool isInf(optix::float3 v)
{
    return isInf(v.x) || isInf(v.y) || isInf(v.z);
}

template<typename T>
RT_FUNCTION void swap(T & t1, T & t2)
{
    T tmp = t1;
    t1 = t2;
    t2 = tmp;
}