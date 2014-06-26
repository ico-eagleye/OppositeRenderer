#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include <cuda_runtime.h>
#include "renderer/RayType.h"
#include "renderer/SubpathPRD.h"

using namespace optix;
using namespace ContextTest;

// From OptiX path_trace sample
//template<unsigned int N>
//static __host__ __device__ __inline__ unsigned int tea( unsigned int val0, unsigned int val1 )
//{
//    unsigned int v0 = val0;
//    unsigned int v1 = val1;
//    unsigned int s0 = 0;
//
//    for( unsigned int n = 0; n < N; n++ )
//    {
//        s0 += 0x9e3779b9;
//        v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
//        v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
//    }
//
//    return v0;
//}

//
//rtDeclareVariable(rtObject, sceneRootObject, , );
//rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
//rtBuffer<uint, 2> lightVertexCountBuffer;
//rtDeclareVariable(uint, lightVertexCountEstimatePass, , );

rtDeclareVariable(rtCallableProgramId<float(float)>, vcmBsdfEvalDiffuse, ,);

RT_PROGRAM void generator()
{
    vcmBsdfEvalDiffuse(1234.f);
    return;
}
//    SubpathPRD lightPrd;
//    lightPrd.depth = 0;
//    lightPrd.keepTracing = 0;
//    lightPrd.done = 0;
//    lightPrd.keepTracing = 0;
//    lightPrd.seed = tea<16>(720u*launchIndex.y+launchIndex.x, 1u);
//    if (lightVertexCountEstimatePass)
//        lightVertexCountBuffer[launchIndex] = 0u;
//
//    float3 rayOrigin = make_float3( 343.0f, 548.0f, 227.0f);
//    float3 rayDirection = make_float3( .0f, -1.0f, .0f);
//    Ray lightRay = Ray(rayOrigin, rayDirection, RayType::LIGHT_VCM, 0.0001, RT_DEFAULT_MAX );
//
//    for (int i=0;;i++)
//    {
//        lightPrd.keepTracing = 0;
//        rtTrace( sceneRootObject, lightRay, lightPrd );
//
//        if (lightPrd.done) 
//        {
//            break;
//        }
//
//        lightRay.origin = lightPrd.origin;
//        lightRay.direction = lightPrd.direction;
//    }
//}


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

RT_CALLABLE_PROGRAM float vcmBsdfEvaluate(float n)
{
    rtPrintf("Callable: %f\n", n);
    return 42.f;
}