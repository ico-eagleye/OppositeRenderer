// TOLICENSE

#pragma once
//#include <optix_world.h>
#include "renderer/BSDF.h"
#include "optix.h"

struct LightVertex
{
    optix::float3   hitPoint;
    optix::float3   throughput;
    LightBSDF       bsdf;
    optix::uint2    launchIndex;    // for debug TODO remove
    optix::uint     pathLen;
    float           dVCM;
    float           dVC;
    float           dVM;
//#if VCM_UNIFORM_VERTEX_SAMPLING
//    float   dVC_unif_vert;
//#endif
};