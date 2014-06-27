#pragma once
#include <optix.h>
#include <optixu/optixu_math_namespace.h>


namespace VcmMeterial
{
    enum E
    {
        DIFFUSE,
        NUM_MATERIALS
    };
}


struct VcmDiffuseBsdf
{
    float3 Kd;
};


struct VcmBsdfData
{
    VcmMeterial::E material;
    union
    {
        VcmDiffuseBsdf bsdfDiffuse;
    };
};