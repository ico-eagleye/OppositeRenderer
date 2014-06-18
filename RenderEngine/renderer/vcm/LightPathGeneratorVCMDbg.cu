/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include <cuda_runtime.h>
#include "config.h"
#include "renderer/RayType.h"
#include "renderer/vcm/SubpathPRD.h"


using namespace optix;

rtDeclareVariable(rtObject, sceneRootObject, , );
rtBuffer<RandomState, 2> randomStates;
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );

RT_PROGRAM void generatorDbg()
{
	SubpathPRD lightPrd;
	lightPrd.depth = 0;
	lightPrd.done = 0;
	lightPrd.randomState = randomStates[launchIndex]; // curand states

	float3 rayOrigin = make_float3( 343.0f, 548.0f, 227.0f);
	float3 rayDirection = make_float3( .0f, -1.0f, .0f);
	Ray lightRay = Ray(rayOrigin, rayDirection, RayType::LIGHT_VCM, 0.0001, RT_DEFAULT_MAX );
	
	for (int i=0;;i++)
	{
		rtTrace( sceneRootObject, lightRay, lightPrd );

		if (lightPrd.done) 
			break;

		lightRay.origin = lightPrd.origin;
		lightRay.direction = lightPrd.direction;
	}

	randomStates[launchIndex] = lightPrd.randomState;
}


// recursive approach
RT_PROGRAM void generatorDbgRC()
{
	SubpathPRD lightPrd;
	lightPrd.depth = 0;
	lightPrd.done = 0;
	lightPrd.randomState = randomStates[launchIndex]; // curand states
	lightPrd.throughput = make_float3(1.0f);

	float3 rayOrigin = make_float3( 343.0f, 548.0f, 227.0f);
	float3 rayDirection = make_float3( .0f, -1.0f, .0f);
	Ray lightRay = Ray(rayOrigin, rayDirection, RayType::LIGHT_VCM, 0.0001, RT_DEFAULT_MAX );
	
	rtTrace( sceneRootObject, lightRay, lightPrd );

	randomStates[launchIndex] = lightPrd.randomState;
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