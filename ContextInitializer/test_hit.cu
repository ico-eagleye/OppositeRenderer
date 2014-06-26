#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "renderer/RayType.h"
#include "renderer/random.h"
#include "renderer/SubpathPRD.h"

using namespace optix;
using namespace ContextTest;

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


rtDeclareVariable(uint2, launchDim, rtLaunchDim, );
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, tHit, rtIntersectionDistance, );
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, ); 

rtDeclareVariable(rtObject, sceneRootObject, , );
rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(SubpathPRD, lightPrd, rtPayload, );

rtDeclareVariable(uint, lightVertexCountEstimatePass, , );
rtBuffer<int> lightVertexBuffer;
rtBuffer<uint, 2> lightVertexCountBuffer;
rtBuffer<uint> lightVertexBufferIndexBuffer; // single element buffer with index for lightVertexBuffer

// NOTE:
// All fail case due setting cosine sampled direction were tested with all rtPrintf statements
// in the generation program commented out
RT_PROGRAM void closestHit()
{
    lightPrd.depth++;

    //if (lightPrd.depth == 2)                                // doesn't prevent crash on second hit
    //{                                                       // when using #1
    //	lightPrd.done = 1;
    //	return;
    //}

    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;

    lightVertexCountBuffer[launchIndex] = lightPrd.depth;

    // Kind of Russian Roulette
    float contProb = luminanceCIE(Kd);
    float rrSample = rnd(lightPrd.seed);                                                  // SDK
    //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - cont %f RR %f \n", contProb, rrSample);
    if (contProb < rrSample)
    {
        lightPrd.done = 1;
        return;
    }

    if (lightVertexCountEstimatePass)
    {
        lightVertexCountBuffer[launchIndex] = lightPrd.depth;
    }
    else
    {
        uint idx = atomicAdd(&lightVertexBufferIndexBuffer[0], 1u);
        //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - store V %u\n", idx);
        lightVertexBuffer[idx] = 1;
    }

    float2 bsdfSample = make_float2(rnd(lightPrd.seed),rnd(lightPrd.seed));
    float3 dir = sampleHemisphereCosOptix(worldShadingNormal, bsdfSample); // --> #1 doesn't work
    //dir = -ray.direction;                                   // --> #2 works (computation in #1 can be left uncommented)
  
    //if (1 < lightPrd.depth)                                 // #1 still causes crash - this shows that crash occurs because of setting
    //    dir = -ray.direction;                               // lightsPrd.direction to cosine sampled direction on first hit

    lightPrd.direction = normalize(dir);     
    lightPrd.origin = hitPoint;

    // #1 doesn't crash if code below uncommented (stop on first hit)
    //if (lightPrd.depth == 1)  // #1 crashes if condition is depth == 2
    //{                         // even though the new direction is never used
    //	lightPrd.done = 1;      // to trace a ray
    //	return;
    //}
}
