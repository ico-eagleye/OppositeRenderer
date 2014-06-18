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
#include "renderer/Light.h"
#include "renderer/ShadowPRD.h"
#include "renderer/RayType.h"
#include "renderer/helpers/helpers.h"
#include "renderer/helpers/samplers.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/light.h"
#include "math/Sphere.h"

#include "renderer/vcm/PathVertex.h"
#include "renderer/vcm/SubpathPRD.h"


using namespace optix;

rtDeclareVariable(rtObject, sceneRootObject, , );
rtBuffer<RandomState, 2> randomStates;
rtBuffer<Light, 1> lights;
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim, );
rtDeclareVariable(Sphere, sceneBoundingSphere, , );

// VCM
rtDeclareVariable(float, misVcWeightFactor, , ); // vmarz TODO set
rtDeclareVariable(float, misVmWeightFactor, , ); // vmarz TODO set

#if ENABLE_RENDER_DEBUG_OUTPUT
rtBuffer<unsigned int, 2> debugPhotonPathLengthBuffer;
#endif

rtBuffer<uint, 2> lightVertexCountBuffer;

RT_PROGRAM void generator()
{
	SubpathPRD lightPrd;
	lightPrd.depth = 0;
	lightPrd.done = 0;
	lightPrd.randomState = randomStates[launchIndex];
	lightPrd.dVC = 0;
	lightPrd.dVM = 0;
	lightPrd.dVCM = 0;
	lightVertexCountBuffer[launchIndex] = 0u;

	// vmarz TODO: pick based on light power
	int lightIndex = 0;
	if (1 < lights.size())
	{
		float sample = getRandomUniformFloat(&lightPrd.randomState);
		lightIndex = intmin((int)(sample*lights.size()), lights.size()-1);
	}

	const Light light = lights[lightIndex];
	const float inverseLightPickPdf = lights.size();

	float3 rayOrigin;
	float3 rayDirection;
	float emissionPdfW;
	float directPdfW;
	float cosAtLight;
	lightPrd.throughput = lightEmit(light, lightPrd.randomState, rayOrigin, rayDirection, emissionPdfW, directPdfW, cosAtLight);
	// vmarz?: do something similar as done for photon, emit towards scene when light far from scene?
	// check if photons normally missing the scene accounted for?

	// Set init data
	emissionPdfW *= inverseLightPickPdf;
	directPdfW *= inverseLightPickPdf;

	lightPrd.throughput /= emissionPdfW;
	//lightPrd.isFinite = isDelta.isFinite ... vmarz?

	lightPrd.dVCM = Mis(directPdfW / emissionPdfW);

	// e.g. if not delta ligth
	//if (!light.isDelta)
	//{
	//	const float usedCosLight = light.isFinite ? cosAtLight : 1.f;
	//	lightPrd.dVC = Mis(usedCosLight / emissionPdfW);
	//}

	lightPrd.dVM = lightPrd.dVC * misVcWeightFactor;

	//dbg
	rayOrigin = make_float3( 343.0f, 548.7999f, 227.0f);
	rayDirection = make_float3( .0f, -1.0f, .0f);

	// Trace
	Ray lightRay = Ray(rayOrigin, rayDirection, RayType::LIGHT_VCM, 0.0001, RT_DEFAULT_MAX );
	
	for (int i=0;;i++)
	{        
		//OPTIX_DEBUG_PRINT(lightPrd.depth, " dir %.2f %.2f %.2f\n",
		   // lightRay.direction.x, lightRay.direction.y, lightRay.direction.z);
		rtTrace( sceneRootObject, lightRay, lightPrd );

		if (lightPrd.done) 
		{
			//OPTIX_DEBUG_PRINT(lightPrd.depth, " done\n");
			break;
		}

		lightRay.origin = lightPrd.origin;
		lightRay.direction = lightPrd.direction;
		//OPTIX_DEBUG_PRINT(lightPrd.depth, "Gen - new org %f %f %f\n", lightRay.origin.x, lightRay.origin.y, lightRay.origin.z);
		//OPTIX_DEBUG_PRINT(lightPrd.depth, "Gen - new org %f %f %f\n", lightRay.direction.x, lightRay.direction.y, lightRay.direction.z);
	}

	randomStates[launchIndex] = lightPrd.randomState;
}


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
		//OPTIX_DEBUG_PRINT(lightPrd.depth, " dir %.2f %.2f %.2f\n",
		//    lightRay.direction.x, lightRay.direction.y, lightRay.direction.z);
		rtTrace( sceneRootObject, lightRay, lightPrd );

		if (lightPrd.done) 
		{
			//lightPrd.done += a;
			//OPTIX_DEBUG_PRINT(lightPrd.depth, " done\n");
			break;
		}

		lightRay.origin = lightPrd.origin;
		lightRay.direction = lightPrd.direction;
		//OPTIX_DEBUG_PRINT(lightPrd.depth, "Gen - new org %f %f %f\n", lightRay.origin.x, lightRay.origin.y, lightRay.origin.z);
		//OPTIX_DEBUG_PRINT(lightPrd.depth, "Gen - new org %f %f %f\n", lightRay.direction.x, lightRay.direction.y, lightRay.direction.z);
	}

	randomStates[launchIndex] = lightPrd.randomState;
}



rtDeclareVariable(SubpathPRD, lightPrd, rtPayload, );
RT_PROGRAM void miss()
{
	//OPTIX_DEBUG_PRINT(lightPrd.depth, " Miss\n");
	lightPrd.done = 1;
}


// Exception handler program
rtDeclareVariable(float3, exceptionErrorColor, , );
RT_PROGRAM void exception()
{
	//rtPrintf("Exception Light ray!\n");
	//rtPrintExceptionDetails();
	lightPrd.throughput = make_float3(0,0,0);
}