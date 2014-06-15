#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include "renderer/RayType.h"
#include "renderer/random.h"
#include "renderer/SubpathPRD.h"

using namespace optix;

rtDeclareVariable(uint2, launchDim, rtLaunchDim, );
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, tHit, rtIntersectionDistance, );
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, ); 

rtDeclareVariable(rtObject, sceneRootObject, , );
rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(SubpathPRD, lightPrd, rtPayload, );


// <From OptiX path_trace sample>
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
// </From OptiX path_trace sample>


RT_PROGRAM void closestHit()
{
  lightPrd.depth++;
  float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
  float3 hitPoint = ray.origin + tHit*ray.direction;

  float hitCosTheta = dot(worldShadingNormal, -ray.direction);
  if (hitCosTheta < 0) return;

  // Russian Roulette
  float contProb = luminanceCIE(Kd);
  float rrSample = rnd(lightPrd.seed);

  if (contProb < rrSample)
  {
    lightPrd.done = 1;
    return;
  }

  float2 bsdfSample = make_float2(rnd(lightPrd.seed),rnd(lightPrd.seed));
  float3 dir = sampleHemisphereCosOptix(worldShadingNormal, bsdfSample);            // doesn't work

  //dir = normalize(2*worldShadingNormal + ray.direction); // works
  //dir = -ray.direction;                                  // works
  lightPrd.direction = normalize(dir);
     
  lightPrd.origin = hitPoint;

  // Doesn't crash if code below uncommented
  //if (lightPrd.depth == 2)
  //{
  //    lightPrd.done = 1;
  //    return;
  //}
}