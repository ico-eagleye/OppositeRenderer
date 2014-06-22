#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include <cuda_runtime.h>
#include "renderer/RayType.h"
#include "renderer/SubpathPRD.h"

using namespace optix;
using namespace ContextTest;

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


rtDeclareVariable(rtObject, sceneRootObject, , );
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtBuffer<uint, 2> lightVertexCountBuffer;
rtBuffer<uint, 2> dbgNoMissHitStops;

// NOTE:
// All rtPrinf fail cases below were tested individually when direction in the ray payload was simply set to
// negation of the incident direction in the closest hit program: lightPrd.direction = -ray.direction
//
// When hemisphere sampling is used, tracing in a LOOP FAILS, but tracing few rays WITHOUT LOOP WORKS (at the bottom)
RT_PROGRAM void generator()
{
    SubpathPRD lightPrd;
    lightPrd.depth = 0;
    lightPrd.keepTracing = 0;
    lightPrd.done = 0;
    lightPrd.keepTracing = 0;
    lightPrd.seed = tea<16>(720u*launchIndex.y+launchIndex.x, 1u);
    lightVertexCountBuffer[launchIndex] = 0u;
    dbgNoMissHitStops[launchIndex] = 0u;

    float3 rayOrigin = make_float3( 343.0f, 548.0f, 227.0f);
    float3 rayDirection = make_float3( .0f, -1.0f, .0f);
    Ray lightRay = Ray(rayOrigin, rayDirection, RayType::LIGHT_VCM, 0.0001, RT_DEFAULT_MAX );

    for (int i=0;;i++)
    {
        lightPrd.keepTracing = 0;
        rtTrace( sceneRootObject, lightRay, lightPrd );

        if (!lightPrd.keepTracing) 
        {
            if (!lightPrd.done)
                dbgNoMissHitStops[launchIndex] = 1;
            break;
        }

        lightRay.origin = lightPrd.origin;
        lightRay.direction = lightPrd.direction;
    }
}


// THIS WORKS with closestHitRecursive
RT_PROGRAM void generatorRecursive()
{
    SubpathPRD lightPrd;
    lightPrd.depth = 0;
    lightPrd.done = 0;
    lightPrd.seed = tea<16>(720u*launchIndex.y+launchIndex.x, 1u);

    // Approx light position in scene (eliminated use of light buffer while debuggin cause for hangs)
    float3 rayOrigin = make_float3( 343.0f, 548.7f, 227.0f);
    float3 rayDirection = normalize(make_float3( .0f, -1.0f, .0f));
    Ray lightRay = Ray(rayOrigin, rayDirection, RayType::LIGHT_VCM, 0.0001, RT_DEFAULT_MAX );	
    int a = launchIndex.x; // left it here to keep same memory layout for local variables as above
    rtTrace( sceneRootObject, lightRay, lightPrd );
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