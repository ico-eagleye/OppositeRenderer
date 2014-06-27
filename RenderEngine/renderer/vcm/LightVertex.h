// TOLICENSE

#pragma once
//#include <optix_world.h>
#include "material/VcmBsdfData.h"
#include "material/BSDF.h"

struct LightVertex
{
    float3 hitPoint;
    float3 throughput;
    BSDF   bsdf;
    float  pathDepth;
    float  dVCM;
    float  dVC;
    float  dVM;
    // bsdf data
    //VcmBsdfData bsdfData;
};