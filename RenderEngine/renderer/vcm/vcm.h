#pragma once
#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "renderer/RayType.h"
#include "renderer/ShadowPRD.h"
#include "renderer/helpers/samplers.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/light.h"
#include "renderer/Light.h"
#include "renderer/Camera.h"
#include "renderer/helpers/helpers.h"
#include "renderer/vcm/SubpathPRD.h"
#include "renderer/vcm/config_vcm.h"



__inline__
__device__ int isOccluded(rtObject &aSeneRootObject, optix::float3 &aPoint, optix::float3 &aDirection, float aTMax)
{
    using namespace optix;
    ShadowPRD shadowPrd;
    shadowPrd.attenuation = 1.0f;
    Ray occlusionRay(aPoint, aDirection, RayType::SHADOW, EPS_RAY, aTMax - 2.f*EPS_RAY);
    rtTrace(aSeneRootObject, occlusionRay, shadowPrd);
    return shadowPrd.attenuation == 0.f;
}