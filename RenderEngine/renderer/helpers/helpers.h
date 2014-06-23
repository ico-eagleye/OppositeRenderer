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
//
// print functions most likely are not the cause it self, some people point to to memory
// corruption issues. It does complicate debugging process, hence the mulptiple switches for the macro below


#define OPTIX_DEBUG_STD_PRINTF 0
#define OPTIX_DEBUG_PRINTF_SPACES 0		// printing multiple consecutive spaces seems ramdom - doesn't always work
#define OPTIX_DEBUG_PRINTF_IDX 1		  // printing multiple consecutive spaces seems ramdom - doesn't always work
#define OPTIX_DEBUG_ID_X 0
#define OPTIX_DEBUG_ID_Y 0


#if OPTIX_DEBUG_STD_PRINTF
#define OPTIX_PRINTF printf
#else
#define OPTIX_PRINTF rtPrintf
#endif

#if ENABLE_RENDER_DEBUG_OUTPUT
#define OPTIX_DEBUG_PRINT(depth, str, ...) \
	if (launchIndex.x == OPTIX_DEBUG_ID_X && launchIndex.y == OPTIX_DEBUG_ID_Y) \
	{  \
		if (OPTIX_DEBUG_PRINTF_IDX) \
		{ \
			if (OPTIX_DEBUG_STD_PRINTF) \
				OPTIX_PRINTF("%d, %d - ", launchIndex.x, launchIndex.y); \
			else \
				OPTIX_PRINTF("%d, %d - d %d - ", launchIndex.x, launchIndex.y, depth); \
		} \
		if (OPTIX_DEBUG_PRINTF_SPACES) for(int i = 0; i < depth; i++) { OPTIX_PRINTF(" "); } \
		OPTIX_PRINTF(str, __VA_ARGS__); \
	}

// original
//#define OPTIX_DEBUG_PRINT(depth, str, ...) \
//	if (launchIndex.x == OPTIX_DEBUG_ID_X && launchIndex.y == OPTIX_DEBUG_ID_Y) \
//	{  \
//	OPTIX_PRINTF("%d %d: ", launchIndex.x, launchIndex.y); \
//	for(int i = 0; i < depth; i++) { OPTIX_PRINTF(" "); } \
//	OPTIX_PRINTF(str, __VA_ARGS__); \
//	}
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

template<typename T>
__device__ __inline__ T sqr(const T& a) { return a*a; }