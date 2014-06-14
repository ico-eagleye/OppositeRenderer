#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include "renderer/RayType.h"
#include "renderer/random.h"
//#include "renderer/helpers/helpers.h"
//#include "renderer/helpers/samplers.h"
#include "renderer/SubpathPRD.h"

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
//rtBuffer<uint, 2> lightVertexCountBuffer;


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



RT_PROGRAM void closestHit()
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
  //lightVertexCountBuffer[launchIndex] = lightPrd.depth;
	
  // Russian Roulette
  float contProb = luminanceCIE(Kd);
  //float rrSample = getRandomUniformFloat(&lightPrd.randomState);
  float rrSample = rnd(lightPrd.seed);
  //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - cont %f RR %f \n", contProb, rrSample);
  if (contProb < rrSample)
  {
    lightPrd.done = 1;
    return;
  }

  //float2 bsdfSample = getRandomUniformFloat2(&lightPrd.randomState);
  float2 bsdfSample = make_float2(rnd(lightPrd.seed),rnd(lightPrd.seed));
  //float3 dir = sampleUnitHemisphereCos(worldShadingNormal, bsdfSample);           // doesn't work
  float3 dir = sampleHemisphereCosOptix(worldShadingNormal, bsdfSample);            // doesn't work
  dir = normalize(dir);
  //if (launchIndex.x == 0 && launchIndex.y == 0)
  //{
  //    OPTIX_DEBUG_PRINT(lightPrd.depth, "%d Hit - samp dir %f %f %f len %f\n", launchIndex.x, 
  //        dir.x, dir.y, dir.z, sqrtf(dot(dir, dir)));	
  //}

  //dir = normalize(2*worldShadingNormal + ray.direction); // works
  //dir = -ray.direction;                                  // works
  lightPrd.direction = normalize(dir);

  //OPTIX_DEBUG_PRINT(lightPrd.depth, " Hit - new dir %f %f %f\n", lightPrd.direction.x, lightPrd.direction.y, lightPrd.direction.z);	
    
  lightPrd.origin = hitPoint;
  //OPTIX_DEBUG_PRINT(lightPrd.depth, " Hit - new org %f %f %f\n", lightPrd.origin.x, lightPrd.origin.y, lightPrd.origin.z);

  // Doesn't crash if code below uncommented
  //if (lightPrd.depth == 2)
  //{
  //    lightPrd.done = 1;
  //    return;
  //}
}