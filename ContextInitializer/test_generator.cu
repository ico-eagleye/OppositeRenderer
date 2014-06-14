#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include <cuda_runtime.h>
#include "renderer/RayType.h"
#include "renderer/SubpathPRD.h"
//#include "renderer/helpers/helpers.h"
//#include "renderer/helpers/samplers.h"

using namespace optix;
using namespace ContextTest;

rtDeclareVariable(rtObject, sceneRootObject, , );
//rtBuffer<RandomState, 2> randomStates;
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );

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


RT_PROGRAM void generator()
{
  	SubpathPRD lightPrd;
	  lightPrd.depth = 0;
    lightPrd.done = 0;
	  //lightPrd.randomState = randomStates[launchIndex]; // curand states
    lightPrd.seed = tea<16>(720*launchIndex.y+launchIndex.x, 1);

    float3 rayOrigin = make_float3( 343.0f, 448.0f, 227.0f);
    float3 rayDirection = normalize(make_float3( .1f, -1.0f, .1f));
    Ray lightRay = Ray(rayOrigin, rayDirection, RayType::LIGHT_VCM, 0.0001, RT_DEFAULT_MAX );
	
    int a = launchIndex.x; // #1 using launchIndex.x doesn't produce same effect

    for (int i=0;;i++)
    {
        // Example 1
        // Without #2 - output in first iteration, then hang (Cuda 6 "Error ir rtPrintf format string")
        // With    #2 - works
        //rtPrintf("Output\n");

        // Example 2
        // Without #2 - output in first iteration, then "Error ir rtPrintf format string"
        // With    #2 - works
        //if (launchIndex.x == 0 && launchIndex.y == 0)
        //{
        //    rtPrintf("Outputs\n");
        //}

        // Example 3
        // Without #2 - works
        // With    #2 - works
        if (launchIndex.x == 0 && launchIndex.y == 0)
        {
            rtPrintf("idx %d i %d\n", launchIndex.x, i);
        }

        rtTrace( sceneRootObject, lightRay, lightPrd );

		    if (lightPrd.done) 
        {
            //lightPrd.done += a; // #2
            break;
        }

        lightRay.origin = lightPrd.origin;
        lightRay.direction = lightPrd.direction;

        // Example 4
        // Without #2 - output in first iteration, then "Error ir rtPrintf format string"
        // With    #2 - works
        //rtPrintf("Output\n");

        //OPTIX_DEBUG_PRINT(lightPrd.depth, "Gen - new org %f %f %f\n", lightRay.origin.x, lightRay.origin.y, lightRay.origin.z);
        //OPTIX_DEBUG_PRINT(lightPrd.depth, "Gen - new org %f %f %f\n", lightRay.direction.x, lightRay.direction.y, lightRay.direction.z);
	}

	//randomStates[launchIndex] = lightPrd.randomState;
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
}