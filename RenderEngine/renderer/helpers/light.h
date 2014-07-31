/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Stian Pedersen
 *                Valdis Vilcans
*/

#pragma once
#define OPTIX_PRINTF_DEF
#define OPTIX_PRINTFI_DEF
#define OPTIX_PRINTFID_DEF

#include "renderer/device_common.h"
#include "renderer/Light.h"
#include "random.h"
#include "helpers.h"
#include "renderer/ShadowPRD.h"
#include "renderer/TransmissionPRD.h"
#include "renderer/helpers/samplers.h"
#include "renderer/vcm/config_vcm.h"
#include "math/Sphere.h"

#define OPTIX_PRINTF_ENABLED 0
#define OPTIX_PRINTFI_ENABLED 0
#define OPTIX_PRINTFID_ENABLED 0

RT_FUNCTION optix::float3 getLightContribution(const Light & light, const optix::float3 & rec_position, 
    const optix::float3 & rec_normal, const rtObject & rootObject, RandomState & randomState)
{
    float lightFactor = 1;

    float lightDistance = 0;
    float3 pointOnLight;

    if(light.lightType == Light::AREA)
    {
        float2 sample = getRandomUniformFloat2(&randomState);
        pointOnLight = light.position + sample.x*light.v1 + sample.y*light.v2;
    }
    else if(light.lightType == Light::POINT)
    {
        pointOnLight = light.position;
        lightFactor *= 1.f/4.f;
    }
    else if(light.lightType == Light::SPOT)
    {
        // Todo find correct direct light for spot light
        lightFactor = 0;
    }

    float3 towardsLight = pointOnLight - rec_position;
    lightDistance = optix::length(towardsLight);
    towardsLight = towardsLight / lightDistance;
    float n_dot_l = maxf(0, optix::dot(rec_normal, towardsLight));
    lightFactor *= n_dot_l / (M_PIf*lightDistance*lightDistance);

    if(light.lightType == Light::AREA)
    {
        lightFactor *= maxf(0, optix::dot(-towardsLight, light.normal));
    }

    if (lightFactor > 0.0f)
    {
        ShadowPRD shadowPrd;
        shadowPrd.attenuation = 1.0f;
        optix::Ray shadow_ray (rec_position, towardsLight, RayType::SHADOW, 0.0001, lightDistance-0.0001);
        rtTrace(rootObject, shadow_ray, shadowPrd);
        lightFactor *= shadowPrd.attenuation;
        

        // Check for participating media transmission
/*#if ENABLE_PARTICIPATING_MEDIA
        TransmissionPRD transmissionPrd;
        transmissionPrd.attenuation = 1.0f;
        optix::Ray transmissionRay (rec_position, towardsLight, RayType::PARTICIPATING_MEDIUM_TRANSMISSION, 0.001, lightDistance-0.01);
        rtTrace(rootObject, transmissionRay, transmissionPrd);
        lightFactor *= transmissionPrd.attenuation;
#endif
        */
        //printf("Point on light:%.2f %.2f %.2f shadowPrd.attenuation %.2f\n", pointOnLight.x, pointOnLight.y, pointOnLight.z, shadowPrd.attenuation);
        return light.power*lightFactor;
    }

    return optix::make_float3(0);
};

#define OPTIX_PRINTFI_ENABLED 0
// Samples emission point and direction, returns particle energy/weight. Fills
// emission and direct hit pdf, cosine at light source
RT_FUNCTION optix::float3 lightEmit(const Sphere & aSceneBoundingSphere, const Light & aLight, RandomState & aRandomState,
                                    float3 & oPosition, float3 & oDirection, float & oEmissionPdfW,
                                    float & oDirectPdfA, float & oCosThetaLight,
                                    optix::uint2 *launchIdx = NULL)
{
    float3 radiance = optix::make_float3(0);
    float2 dirRnd = getRandomUniformFloat2(&aRandomState);

    if(aLight.lightType == Light::AREA)
    {
        float2 posRnd = getRandomUniformFloat2(&aRandomState);
        oPosition = aLight.position + posRnd.x*aLight.v1 + posRnd.y*aLight.v2;
        // cannot not emit particle so bias direction if cosine is too low (true last parameter)
        oDirection = sampleUnitHemisphereCos(aLight.normal, dirRnd, &oEmissionPdfW, &oCosThetaLight, true);
        oEmissionPdfW *= aLight.inverseArea; // p0_connect * p1 // for [tech. rep. (31)]
        oDirectPdfA = aLight.inverseArea;    // p0_direct
        radiance = aLight.Lemit * oCosThetaLight;
    }
    else if(aLight.lightType == Light::POINT)
    {
        oPosition = aLight.position;
        //oDirection = sampleUnitSphere(dirRnd, &oEmissionPdfW);
        
        float3 lightToSceneCenter = aSceneBoundingSphere.center.__of3 - aLight.position;
        float lightSceneCenterDistance = optix::length(lightToSceneCenter);
        lightToSceneCenter /= lightSceneCenterDistance;
        bool lightOutsideSphere = aSceneBoundingSphere.radius < lightSceneCenterDistance;
        if (lightOutsideSphere)
        {
            // angle between dir to to center and sphere tangent
            float theta = asinf(aSceneBoundingSphere.radius / lightSceneCenterDistance);
            oDirection = sampleCone(dirRnd, theta, lightToSceneCenter, &oEmissionPdfW);
        }
        else
        {
            oDirection = sampleUnitSphere(dirRnd, &oEmissionPdfW);
        }
#if DEBUG_EMIT_DIR_FIXED
        oDirection = DEBUG_EMIT_DIR;
#endif
        oDirectPdfA = 1.f;
        oCosThetaLight = 1.f;           // not used for delta lights
        radiance = aLight.intensity;
    }
    else if(aLight.lightType == Light::SPOT)
    {
        // Todo find correct direct light for spot light
    }
    
    return radiance;
};

#define OPTIX_PRINTFI_ENABLED 0
// Samples emission point on light source, returns radiance towards receiving point. Fills
// emission and direct hit pdf, cosine at light source
RT_FUNCTION optix::float3 lightIlluminate(const Sphere & aSceneBoundingSphere, const Light & aLight, RandomState & aRandomState, 
                                          const float3 & aReceivePosition, float3 & oDirectionToLight, float & oDistance,
                                          float & oDirectPdfW, float * oEmissionPdfW = NULL,
                                          float * oCosThetaLight = NULL, optix::uint2 *launchIdx = NULL)
{
    using namespace optix;    
    float3 radiance = optix::make_float3(0);

    if (aLight.lightType == Light::AREA)
    {
        float2 posRnd = getRandomUniformFloat2(&aRandomState);
        float3 pointOnLight = aLight.position + posRnd.x*aLight.v1 + posRnd.y*aLight.v2;
        oDirectionToLight = pointOnLight - aReceivePosition;
        oDistance = length(oDirectionToLight);
        oDirectionToLight /= oDistance;

        float cosThetaLight = dot(aLight.normal, -oDirectionToLight);
        if (cosThetaLight < EPS_COSINE)
            return radiance;

        // convert area sampling pdf mInvArea to pdf w.r.t solid angle 
        // (mult with inverse of W to A conversion factor cos/distSqr)
        oDirectPdfW = aLight.inverseArea * sqr(oDistance) / cosThetaLight;
        if(oCosThetaLight)
            *oCosThetaLight = cosThetaLight;

        if(oEmissionPdfW)
            *oEmissionPdfW = aLight.inverseArea * cosThetaLight * M_1_PIf;

        radiance = aLight.Lemit;
        return radiance;
    }
    else if(aLight.lightType == Light::POINT)
    {
        oDirectionToLight = aLight.position - aReceivePosition;
        oDistance = length(oDirectionToLight);
        oDirectionToLight /= oDistance;

        float3 lightToSceneCenter = aSceneBoundingSphere.center.__of3 - aLight.position;
        float lightSceneCenterDistance = optix::length(lightToSceneCenter);
        lightToSceneCenter /= lightSceneCenterDistance;
        bool lightOutsideSphere = aSceneBoundingSphere.radius < lightSceneCenterDistance;        
        if (lightOutsideSphere)
        {
            float theta = asinf(aSceneBoundingSphere.radius / lightSceneCenterDistance);
            if (oEmissionPdfW)
                *oEmissionPdfW = sampleConePdfW(theta);
        }
        else
        {

            if (oEmissionPdfW)
                *oEmissionPdfW = 0.25f * M_1_PIf; // uniform sphere sampling pdf
        }

        oDirectPdfW = sqr(oDistance);
        if (oCosThetaLight)
            *oCosThetaLight = 1.f;
        radiance = aLight.intensity;
    }
    else if(aLight.lightType == Light::SPOT)
    {
        // Todo find correct direct light for spot light
    }

    return radiance;
}

#undef OPTIX_PRINTF_ENABLED
#undef OPTIX_PRINTFI_ENABLED
#undef OPTIX_PRINTFID_ENABLED