#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include <cuda_runtime.h>
#include "renderer/RayType.h"
#include "renderer/SubpathPRD.h"

using namespace optix;
using namespace ContextTest;

rtDeclareVariable(rtObject, sceneRootObject, , );
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );

// From OptiX path_trace sample
template<unsigned int N>
static __host__ __device__ __inline__ unsigned int tea( unsigned int val0, unsigned int val1 )
{
	unsigned int v0 = val0;
	unsigned int v1 = val1;
	unsigned int s0 = 0;

	for( unsigned int n = 0; n < N; n++ )
	{
		s0 += 0x9e3779b9;
		v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
		v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
	}

	return v0;
}

// NOTE:
// All rtPrinf fail cases below were tested individually when direction in the ray payload was simply set to
// negation of the incident direction in the closest hit program: lightPrd.direction = -ray.direction
//
// When hemisphere sampling is used, tracing in a LOOP FAILS, but tracing few rays WITHOUT LOOP WORKS (at the bottom)
RT_PROGRAM void generator()
{
	SubpathPRD lightPrd;
	lightPrd.depth = 0;
	lightPrd.done = 0;
	lightPrd.seed = tea<16>(720u*launchIndex.y+launchIndex.x, 1u);

	// Approx light position in scene (eliminated use of light buffer while debuggin cause for hangs)
	float3 rayOrigin = make_float3( 343.0f, 548.7f, 227.0f);

	// !!! Use of cosine sampling in closet hit program
	// - WORKS if init unnormalized dir here is something like ( .0f, -1.0f, 1.0f)
	// - FAILS if init unnormalized dir here is something like ( 1.0f, -1.0f, .0f)
	float3 rayDirection = normalize(make_float3( .0f, -1.0f, .0f));
	Ray lightRay = Ray(rayOrigin, rayDirection, RayType::LIGHT_VCM, 0.0001, RT_DEFAULT_MAX );
	
	int a = launchIndex.x;
	// Optix 3.5.1 specific - when line #1 was uncommented examples 1, 2, 5 (example 4 not tested)
	// Using launchIndex.x withing the loop instead of 'a' didn't produce same effect

	// FAILS in second interation if lightPrd.direction computed by sampling hemisphere
	for (int i=0;;i++)
	{
		// Example 1 - FAILS
		// output in first iteration, then hangs (Cuda 5.5) or "Error ir rtPrintf format string" (Cuda 6)
		//rtPrintf("Output\n");

		// Example 2 - FAILS
		// output in first iteration, then "Error ir rtPrintf format string"
		//if (launchIndex.x == 0 && launchIndex.y == 0) rtPrintf("Outputs\n");

		// Example 3 - WORKS ( ALL EXAMPLES WORK TOGETHER WITH THIS !!! )
		// note that launchIndex.x is passed to rtPrintf()
		//if (launchIndex.x == 0 && launchIndex.y == 0) rtPrintf("idx %d iteration i %d\n", launchIndex.x, i);

		rtTrace( sceneRootObject, lightRay, lightPrd );

		if (lightPrd.done)
		{
			//lightPrd.done += a;	// --> #1
			break;
		}

		// !!! If ray updates below are commented out then example 3 also fails in second iteration.
		// As mentioned, all examples work together with 3, so example 4 outputs in first iteration and then example 3 fails in second.
		lightRay.origin = lightPrd.origin;
		lightRay.direction = lightPrd.direction;

		// creating new ray doesn't help to avoid crash when cosine sampled direction set in closest hit program
		//lightRay = Ray(lightPrd.origin, lightPrd.direction, RayType::LIGHT_VCM, 0.0001, RT_DEFAULT_MAX );

		// Example 4 - FAILS
		// output in first iteration, then "Error ir rtPrintf format string"
		//if (launchIndex.x == 0 && launchIndex.y == 0) rtPrintf("idx %d iteration i %d prepared new ray\n", launchIndex.x, i);

		// Example 5 - FAILS
		// output in first iteration, then "Error ir rtPrintf format string"
		//rtPrintf("Output\n");
	}

	// TRACING WITHOUT LOOP - WORKS
	//
	// Works with hemisphere sampled direction BUT "new line" characters is ignored, everything printed in one line
	//rtTrace( sceneRootObject, lightRay, lightPrd );
	//if (launchIndex.x == 0 && launchIndex.y == 0) rtPrintf("idx %d Traced 1 END %d \n", launchIndex.x);

	//lightRay.origin = lightPrd.origin;
	//lightRay.direction = lightPrd.direction;
	//rtTrace( sceneRootObject, lightRay, lightPrd );
	//if (launchIndex.x == 0 && launchIndex.y == 0) rtPrintf("idx %d Traced 2 %d \n", launchIndex.x);

	//lightRay.origin = lightPrd.origin;
	//lightRay.direction = lightPrd.direction;
	//rtTrace( sceneRootObject, lightRay, lightPrd );
	//if (launchIndex.x == 0 && launchIndex.y == 0) rtPrintf("idx %d Traced 3 %d \n", launchIndex.x);
}


rtDeclareVariable(SubpathPRD, lightPrd, rtPayload, );
RT_PROGRAM void miss()
{
	lightPrd.done = 1;
}


// Exception handler program
RT_PROGRAM void exception()
{
	rtPrintf("Exception Light ray!\n");
	rtPrintExceptionDetails();
}