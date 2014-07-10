/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "renderer/RadiancePRD.h"
#include "renderer/ShadowPRD.h"
#include "renderer/ppm/PhotonPRD.h"
#include "renderer/vcm/SubpathPRD.h"
#include "renderer/vcm/config_vcm.h"
#include "renderer/vcm/vcm.h"

using namespace optix;

rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, tHit, rtIntersectionDistance, );
rtDeclareVariable(RadiancePRD, radiancePrd, rtPayload, );
rtDeclareVariable(float3, powerPerArea, , );
rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(ShadowPRD, shadowPrd, rtPayload, );
rtDeclareVariable(PhotonPRD, photonPrd, rtPayload, );

/*
// Radiance Program
*/
RT_PROGRAM void closestHitRadiance()
{
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    float3 Le = powerPerArea/M_PIf;
    radiancePrd.radiance += radiancePrd.attenuation*Le;
    radiancePrd.flags |= PRD_HIT_EMITTER;
    radiancePrd.lastTHit = tHit;
}

/*
// Photon Program
*/
RT_PROGRAM void closestHitPhoton()
{
   photonPrd.depth++;
}


// Radiance shadow program
RT_PROGRAM void gatherAnyHitOnEmitter()
{
    shadowPrd.attenuation = 1.0f;
    rtTerminateRay();
}



rtDeclareVariable(SubpathPRD, subpathPrd, rtPayload, );

/*
// VCM Programs
*/
RT_PROGRAM void closestHitLight()
{
    subpathPrd.done = 1;
}

#define OPTIX_PRINTF_ENABLED 1
#define OPTIX_PRINTFI_ENABLED 1
#define OPTIX_PRINTFID_ENABLED 1

rtDeclareVariable(float3, Lemit, , );
rtDeclareVariable(float, inverseArea, , );
rtDeclareVariable(rtObject, sceneRootObject, , );
rtBuffer<Light, 1> lights;
rtDeclareVariable(int, vcmUseVC, , );
rtDeclareVariable(int, vcmUseVM, , );

RT_PROGRAM void vcmClosestHitCamera()
{
    subpathPrd.depth++;
    subpathPrd.done = 1;
    if (isZero(Lemit)) 
        return;

    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;

    float lightPickProb = 1.f / lights.size();
  
    float cosAtLight = maxf(0.f, dot(worldShadingNormal, -ray.direction));
    if (cosAtLight == 0.f) 
        return;

    float directPdfA = inverseArea;
    float emissionPdfW = inverseArea * CosHemispherePdfW(worldShadingNormal, -ray.direction);

    connectLightSourceS0(subpathPrd, Lemit, directPdfA, emissionPdfW, lightPickProb, vcmUseVC, vcmUseVM); 
}

