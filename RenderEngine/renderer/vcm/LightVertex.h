// TOLICENSE

#pragma once
//#include <optix_world.h>
#include "material/BSDF.h"

struct LightVertex
{
    float3  hitPoint;
    float3  throughput;
    VcmBSDF bsdf;
    float   pathDepth;
    float   dVCM;
    float   dVC;
    float   dVM;
};