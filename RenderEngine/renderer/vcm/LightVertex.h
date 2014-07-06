// TOLICENSE

#pragma once
//#include <optix_world.h>
#include "material/BSDF.h"

struct LightVertex
{
    float3  hitPoint;
    float3  throughput;
    VcmBSDF bsdf;
    uint2   launchIndex;    // for debug TODO remove
    float   pathDepth;
    float   dVCM;
    float   dVC;
    float   dVM;
//#if VCM_UNIFORM_VERTEX_SAMPLING
//    float   dVC_unif_vert;
//#endif
};