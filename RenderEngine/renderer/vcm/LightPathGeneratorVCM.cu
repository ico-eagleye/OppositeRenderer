/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include <optix.h>
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
#include "renderer/ppm/Photon.h"
#include "renderer/ppm/PhotonPRD.h"
#include "math/Sphere.h"

#include "renderer/vcm/PathVertex.h"
#include "renderer/vcm/SubpathPRD.h"


using namespace optix;

rtDeclareVariable(rtObject, sceneRootObject, , );
rtBuffer<Photon, 1> photons;
rtBuffer<RandomState, 2> randomStates;
rtDeclareVariable(uint, maxPhotonDepositsPerEmitted, , );
rtDeclareVariable(uint, photonLaunchWidth, , );
rtBuffer<Light, 1> lights;
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim, );
rtDeclareVariable(Sphere, sceneBoundingSphere, , );

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


RT_PROGRAM void generator()
{
	SubpathPRD lightPRD;
	lightPRD.depth = 0;
	lightPRD.randomState = randomStates[launchIndex];
	lightPRD.dVC = 0;
	lightPRD.dVM = 0;
	lightPRD.dVCM = 0;

	// vmarz?: pick based on light power?
	int lightIndex = 0;
	if (1 < lights.size())
	{
		float sample = getRandomUniformFloat(&lightPRD.randomState);
		lightIndex = intmin((int)(sample*lights.size()), lights.size()-1);
	}

	const Light light = lights[lightIndex];
	const float inverseLightPickPdf = lights.size();

	float3 rayOrigin, rayDirection;
	float emissionPdfW, directPdfW, cosAtLight;
	lightPRD.throughput = lightEmit(light, lightPRD.randomState, rayOrigin, rayDirection, emissionPdfW, directPdfW, cosAtLight);
	// vmarz?: do something similar as done for photon, emit towards scene when light far from scene?
	// check if photons normally missing the scene accounted for?

	emissionPdfW *= inverseLightPickPdf;
	directPdfW *= inverseLightPickPdf;

	lightPRD.throughput /= emissionPdfW;
	//lightPRD.isFinite = isDelta.isFinite ... vmarz?

	// e.g. if not delta ligth
	if (!light.isDelta)
	{
		const float usedCosLight = light.isFinite ? cosAtLight : 1.f;
		lightPRD.dVC = Mis(usedCosLight / emissionPdfW);
	}

	//lightPRD.dVM = lightPRD.dVC * mMisVcWeightFactor; // vmarz: TODO

	Ray lightRay = Ray(rayOrigin, rayDirection, RayType::LIGHT_VCM, 0.0001, RT_DEFAULT_MAX );
	rtTrace( sceneRootObject, lightRay, lightPRD );

	randomStates[launchIndex] = lightPRD.randomState;

#if ENABLE_RENDER_DEBUG_OUTPUT
	debugPhotonPathLengthBuffer[launchIndex] = lightPRD.depth;
#endif
}

rtDeclareVariable(SubpathPRD, lightPRD, rtPayload, );
RT_PROGRAM void miss()
{
    OPTIX_DEBUG_PRINT(lightPRD.depth, "Light ray missed geometry.\n");
}

//
// Exception handler program
//
rtDeclareVariable(float3, exceptionErrorColor, , );
RT_PROGRAM void exception()
{
    printf("Exception Light ray!\n");
    lightPRD.throughput = make_float3(0,0,0);
}