/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include "config.h"
#include "renderer/Hitpoint.h"
#include "renderer/RayType.h"
#include "renderer/RadiancePRD.h"
#include "renderer/ppm/PhotonPRD.h"
#include "renderer/ppm/Photon.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/helpers.h"
#include "renderer/helpers/samplers.h"
#include "renderer/helpers/store_photon.h"
#include "renderer/vcm/SubpathPRD.h"
#include "renderer/vcm/PathVertex.h"

using namespace optix;

rtDeclareVariable(uint2, launchDim, rtLaunchDim, );
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, tHit, rtIntersectionDistance, );

rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, ); 
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, ); 

rtDeclareVariable(rtObject, sceneRootObject, , );
rtDeclareVariable(float3, Kd, , );

rtDeclareVariable(SubpathPRD, lightPrd, rtPayload, );
rtBuffer<uint, 2> lightVertexCountBuffer;

RT_PROGRAM void closestHitLightDbg()
{
	lightPrd.depth++;    
	if (0.5f < getRandomUniformFloat(&lightPrd.randomState))
	{
		lightPrd.done = 1;
		return;
	}

	//float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
	float3 hitPoint = ray.origin + tHit*ray.direction;
	lightPrd.origin = hitPoint;
	lightPrd.direction = -ray.direction;
	//OPTIX_DEBUG_PRINT(lightPrd.depth, " Hit - point %f %f %f\n", hitPoint.x, hitPoint.y, hitPoint.z);

	//float hitCosTheta = dot(worldShadingNormal, -ray.direction);
	//if (hitCosTheta < 0) return;
	//OPTIX_DEBUG_PRINT(lightPrd.depth, " Hit - cos theta %f \n", hitCosTheta);

	// store path vertex
	//lightVertexCountBuffer[launchIndex] = lightPrd.depth;
	
	// Russian Roulette
	//float contProb = luminanceCIE(Kd);
	//float rrSample = getRandomUniformFloat(&lightPrd.randomState);    
	//OPTIX_DEBUG_PRINT(lightPrd.depth, " Hit - cont %f RR %f \n", contProb, rrSample);
	//if (0.5f < rrSample)
	//{
	//	lightPrd.done = 1;
	//	return;
	//}

	// New dir
	//OPTIX_DEBUG_PRINT(lightPrd.depth, " Hit - new dir %f %f %f\n", lightPrd.direction.x, lightPrd.direction.y, lightPrd.direction.z);	
	//OPTIX_DEBUG_PRINT(lightPrd.depth, " Hit - new org %f %f %f\n", lightPrd.origin.x, lightPrd.origin.y, lightPrd.origin.z);

	// Doesn't crash if code below uncommented
	//if (lightPrd.depth == 1)
	//{
	//    lightPrd.done = 1;
	//    return;
	//}
}


// Create ONB from normalaized vector
static __device__ __inline__ void createONB( 
	const optix::float3& n, optix::float3& U, optix::float3& V)
{
  using namespace optix;

  U = cross( n, make_float3( 0.0f, 1.0f, 0.0f ) );
  if ( dot(U, U) < 1.e-3f )
	  U = cross( n, make_float3( 1.0f, 0.0f, 0.0f ) );
  U = normalize( U );
  V = cross( n, U );
}


float3 __device__ __inline__ sampleHemisphereCosOptix(float3 normal, float2 rnd)
{
	float3 p;
	cosine_sample_hemisphere(rnd.x, rnd.y, p);
	float3 v1, v2;
	createONB(normal, v1, v2);
	return v1 * p.x + v2 * p.y + normal * p.z;  
}


RT_PROGRAM void closestHitLightDbgRC()
{
	lightPrd.depth++;

	float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
	float3 hitPoint = ray.origin + tHit*ray.direction;

	//OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - rayDir %f %f %f\n", ray.direction.x, ray.direction.y, ray.direction.z);
	//OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - point %f %f %f\n", hitPoint.x, hitPoint.y, hitPoint.z);
	//OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - normal %f %f %f\n", worldShadingNormal.x, worldShadingNormal.y, worldShadingNormal.z);

	float hitCosTheta = dot(worldShadingNormal, -ray.direction);
	if (hitCosTheta < 0) return;
	//OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - cos theta %f \n", hitCosTheta);

	// store path vertex
	lightVertexCountBuffer[launchIndex] = lightPrd.depth;
	
	// Russian Roulette
	float contProb = luminanceCIE(Kd);
	float rrSample = getRandomUniformFloat(&lightPrd.randomState);    
	//OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - cont %f RR %f \n", contProb, rrSample);
	if (contProb < rrSample)
	{
		lightPrd.done = 1;
		return;
	}
	lightPrd.throughput /= contProb;

	float2 bsdfSample = getRandomUniformFloat2(&lightPrd.randomState);
	float3 dir = sampleUnitHemisphereCos(worldShadingNormal, bsdfSample);
	dir = sampleHemisphereCosOptix(worldShadingNormal, bsdfSample);
	//OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - samp dir %f %f %f len %f\n", dir.x, dir.y, dir.z, sqrtf(dot(dir, dir)));	
	//lightPrd.direction = normalize(dir);
 //   lightPrd.direction = normalize(4*worldShadingNormal + ray.direction);

	//OPTIX_DEBUG_PRINT(lightPrd.depth, " Hit - new dir %f %f %f\n", lightPrd.direction.x, lightPrd.direction.y, lightPrd.direction.z);	
	dir = normalize(dir);

	Ray newRay = make_Ray(hitPoint, dir, RayType::LIGHT_VCM, RAY_LEN_MIN, RT_DEFAULT_MAX);
	rtTrace( sceneRootObject, newRay, lightPrd );

	//OPTIX_DEBUG_PRINT(lightPrd.depth, " Hit - new org %f %f %f\n", lightPrd.origin.x, lightPrd.origin.y, lightPrd.origin.z);

	// Doesn't crash if code below uncommented
	//if (lightPrd.depth == 2)
	//{
	//    lightPrd.done = 1;
	//    return;
	//}
}