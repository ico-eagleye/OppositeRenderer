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

//optix::float3 __inline __device__ lightEmit(const Light & aLight, RandomState & aRandomState,
//											float3 & oPosition, float3 & oDirection, float & oEmissionPdfW,
//											float & oDirectPdfA, float & oCosThetaLight)
//struct SubpathPRD
//{
//    optix::float3 origin;
//	optix::float3 direction;
//	optix::float3 throughput;
//    optix::uint depth;
//    RandomState randomState;
//	float dVCM;
//	float dVC;
//	float dVM;
//	//uint  mIsFiniteLight :  1; // Just generate by finite light
//    //uint  mSpecularPath  :  1; // All scattering events so far were specular
//};

rtBuffer<ushort, 2> lightVertexCountBuffer;

RT_PROGRAM void generator()
{
	SubpathPRD lightPrd;
	//lightPrd.depth = 0;
    lightPrd.done = 0;
	//lightPrd.randomState = randomStates[launchIndex];
	//lightPrd.dVC = 0;
	//lightPrd.dVM = 0;
	//lightPrd.dVCM = 0;
    //lightVertexCountBuffer[launchIndex] = lightPrd.depth;

	// vmarz?: pick based on light power?
	//int lightIndex = 0;
	//if (1 < lights.size())
	//{
	//	float sample = getRandomUniformFloat(&lightPrd.randomState);
	//	lightIndex = intmin((int)(sample*lights.size()), lights.size()-1);
	//}

	//const Light light = lights[lightIndex];
	//const float inverseLightPickPdf = lights.size();

    const Light light = lights[0];
	float3 rayOrigin = make_float3( 343.0f, 548.7999f, 227.0f);
	float3 rayDirection = make_float3( .0f, -1.0f, .0f);
	//float emissionPdfW;
	//float directPdfW;
	//float cosAtLight;
	//lightPrd.throughput = lightEmit(light, lightPrd.randomState, rayOrigin, rayDirection, emissionPdfW, directPdfW, cosAtLight);
	//// vmarz?: do something similar as done for photon, emit towards scene when light far from scene?
	//// check if photons normally missing the scene accounted for?
	//
	//// Set init data
	//emissionPdfW *= inverseLightPickPdf;
	//directPdfW *= inverseLightPickPdf;

	//lightPrd.throughput /= emissionPdfW;
	//lightPrd.isFinite = isDelta.isFinite ... vmarz?

	//lightPrd.dVCM = Mis(directPdfW / emissionPdfW);

	// e.g. if not delta ligth
	//if (!light.isDelta)
	//{
	//	const float usedCosLight = light.isFinite ? cosAtLight : 1.f;
	//	lightPrd.dVC = Mis(usedCosLight / emissionPdfW);
	//}

	//lightPrd.dVM = lightPrd.dVC * misVcWeightFactor;

	// Trace
	Ray lightRay = Ray(rayOrigin, rayDirection, RayType::LIGHT_VCM, 0.0001, RT_DEFAULT_MAX );
	
    for (int i=0;;i++)
	{
        if ((launchIndex.x + launchIndex.y) != 0)
        {
            printf("Gen %d - idx %d,%d - break\n", i, launchIndex.x, launchIndex.y);
            break;
        }
        
        printf("Gen %d - idx %d,%d - Dir %f %f %f\n", i, launchIndex.x, launchIndex.y, 
            rayDirection.x, rayDirection.y, rayDirection.z);
        rtTrace( sceneRootObject, lightRay, lightPrd );

		if (lightPrd.done) 
        {
            printf("Gen %d - idx %d,%d - break\n", i, launchIndex.x, launchIndex.y);
            break;
        }
        //else
        //    lightPrd.done = 1;

        lightRay.origin = lightPrd.origin;
        lightRay.direction = lightPrd.direction;
        printf("Gen %d - idx %d,%d - isdone %d \n", i, launchIndex.x, launchIndex.y, i, lightPrd.done);
	}

	randomStates[launchIndex] = lightPrd.randomState;
    printf("Done idx %d,%d \n", launchIndex.x, launchIndex.y);
}


//RT_PROGRAM void generator()
//{
//	SubpathPRD lightPrd;
//	lightPrd.depth = 0;
//	lightPrd.randomState = randomStates[launchIndex];
//	lightPrd.dVC = 0;
//	lightPrd.dVM = 0;
//	lightPrd.dVCM = 0;
//
//	// vmarz?: pick based on light power?
//	int lightIndex = 0;
//	if (1 < lights.size())
//	{
//		float sample = getRandomUniformFloat(&lightPrd.randomState);
//		lightIndex = intmin((int)(sample*lights.size()), lights.size()-1);
//	}
//
//	const Light light = lights[lightIndex];
//	const float inverseLightPickPdf = lights.size();
//
//	float3 rayOrigin;
//	float3 rayDirection;
//	float emissionPdfW;
//	float directPdfW;
//	float cosAtLight;
//	lightPrd.throughput = lightEmit(light, lightPrd.randomState, rayOrigin, rayDirection, emissionPdfW, directPdfW, cosAtLight);
//	// vmarz?: do something similar as done for photon, emit towards scene when light far from scene?
//	// check if photons normally missing the scene accounted for?
//	
//	// Set init data
//	emissionPdfW *= inverseLightPickPdf;
//	directPdfW *= inverseLightPickPdf;
//
//	lightPrd.throughput /= emissionPdfW;
//	//lightPrd.isFinite = isDelta.isFinite ... vmarz?
//
//	lightPrd.dVCM = Mis(directPdfW / emissionPdfW);
//
//	// e.g. if not delta ligth
//	if (!light.isDelta)
//	{
//		const float usedCosLight = light.isFinite ? cosAtLight : 1.f;
//		lightPrd.dVC = Mis(usedCosLight / emissionPdfW);
//	}
//
//	lightPrd.dVM = lightPrd.dVC * misVcWeightFactor;
//
//	// Trace
//	Ray lightRay = Ray(rayOrigin, rayDirection, RayType::LIGHT_VCM, 0.0001, RT_DEFAULT_MAX );
//	rtTrace( sceneRootObject, lightRay, lightPrd );
//
//	randomStates[launchIndex] = lightPrd.randomState;
//
//#if ENABLE_RENDER_DEBUG_OUTPUT
//	debugPhotonPathLengthBuffer[launchIndex] = lightPrd.depth;
//#endif
//}


rtDeclareVariable(SubpathPRD, lightPrd, rtPayload, );
RT_PROGRAM void miss()
{
    printf("Miss %d,%d - Dep %d - done\n", launchIndex.x, launchIndex.y, lightPrd.depth);
    lightPrd.done = 1;
    OPTIX_DEBUG_PRINT(lightPrd.depth, "Light ray missed geometry.\n");
}


// Exception handler program
rtDeclareVariable(float3, exceptionErrorColor, , );
RT_PROGRAM void exception()
{
    printf("Exception Light ray!\n");
    rtPrintExceptionDetails();
    lightPrd.throughput = make_float3(0,0,0);
}