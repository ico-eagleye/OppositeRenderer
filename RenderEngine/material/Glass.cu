/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#define OPTIX_PRINTF_DEF
#define OPTIX_PRINTFI_DEF
#define OPTIX_PRINTFID_DEF
#define OPTIX_PRINTFC_DEF
#define OPTIX_PRINTFCID_DEF

#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "config.h"
#include "renderer/Hitpoint.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/helpers.h"
#include "renderer/RayType.h"
#include "renderer/RadiancePRD.h"
#include "renderer/ppm/PhotonPRD.h"
#include "renderer/vcm/vcm.h"
#include "renderer/vcm/LightVertex.h"
#include "renderer/vcm/SubpathPRD.h"
#include "renderer/Light.h"

#define OPTIX_PRINTF_ENABLED 0
#define OPTIX_PRINTFI_ENABLED 0
#define OPTIX_PRINTFID_ENABLED 0
#define OPTIX_PRINTFC_ENABLED 0
#define OPTIX_PRINTFCID_ENABLED 0

using namespace optix;

// Scene wide variables
rtDeclareVariable(rtObject, sceneRootObject, , );

// Ray generation program
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );

// Closest hit material
rtDeclareVariable(float3, Kr, , );
rtDeclareVariable(float, indexOfRefraction, , );
rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, ); 
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, ); 
rtDeclareVariable(RadiancePRD, radiancePrd, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, tHit, rtIntersectionDistance, );

struct IndexOfRefractions
{
    float n1;
    float n2;
};

__device__ __inline bool willTravelInsideGlass (bool hitFromOutside, bool reflection)
{
    return hitFromOutside && !reflection || !hitFromOutside && reflection;
}

__device__ __inline IndexOfRefractions getIndexOfRefractions(bool hitFromOutside, float glassIOR)
{
    IndexOfRefractions i;
    
    if(hitFromOutside)
    {
        i.n1 = 1;
        i.n2 = glassIOR;
    }
    else
    {
        i.n1 = glassIOR;
        i.n2 = 1;
    }
    return i;
}

__device__ __inline float reflectionFactor(float cosI, float cosT, float n1, float n2)
{
    float rp = (n2*cosI - n1*cosT)/(n2*cosI+n1*cosT);
    float rs = (n1*cosI - n2*cosT)/(n1*cosI+n2*cosT);
    return ( rp*rp + rs*rs ) / 2.f ;
}

RT_PROGRAM void closestHitRadiance()
{
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    bool isHitFromOutside = hitFromOutside(ray.direction, worldShadingNormal);
    float3 N = isHitFromOutside ? worldShadingNormal : -worldShadingNormal;
    float3 hitPoint = ray.origin + tHit*ray.direction;

    //OPTIX_PRINTFI("Radiance hit glass %s P(%.2f %.2f %.2f)\n", isHitFromOutside ? "outside" : "inside", hitPoint.x, hitPoint.y, hitPoint.z);

    IndexOfRefractions ior = getIndexOfRefractions(isHitFromOutside, indexOfRefraction);
    float3 refractionDirection;
    bool validRefraction = refract(refractionDirection, ray.direction, N, ior.n2/ior.n1);
    float cosThetaI = -dot(ray.direction, N);
    float cosThetaT = -dot(refractionDirection, N);

    // Find reflection factor using Fresnel equation
    float reflFactor = validRefraction ? reflectionFactor(cosThetaI, cosThetaT, ior.n1, ior.n2) : 1.f;

    float sample = getRandomUniformFloat(&radiancePrd.randomState);
    bool isReflected = (sample <= reflFactor);
    float3 newRayDirection;

    if(isReflected)
    {
        newRayDirection = reflect(ray.direction, N);
    }
    else
    {
        newRayDirection = refractionDirection;
        radiancePrd.attenuation *= (ior.n2*ior.n2)/(ior.n1*ior.n1);
    }

    radiancePrd.flags |= PRD_HIT_SPECULAR;
    radiancePrd.flags &= ~PRD_HIT_NON_SPECULAR;
    
    radiancePrd.depth++;
    // If we will travel inside the glass object, we set type to be RayType::RADIANCE_IN_PARTICIPATING_MEDIUM to avoid intersecting the 
    // participating media

    bool travellingInside = willTravelInsideGlass(isHitFromOutside, isReflected);
    RayType::E rayType = travellingInside ? RayType::RADIANCE_IN_PARTICIPATING_MEDIUM : RayType::RADIANCE;
    Ray newRay = Ray(hitPoint, newRayDirection, rayType, 0.0001, RT_DEFAULT_MAX );

    if(radiancePrd.depth <= MAX_RADIANCE_TRACE_DEPTH)
    {
        rtTrace( sceneRootObject, newRay, radiancePrd );
    }
    else
    {
        radiancePrd.attenuation *= 0;
    }

    radiancePrd.lastTHit = tHit;
}


RT_PROGRAM void anyHitRadiance()
{
    float3 worldShadingNormal = rtTransformNormal(RT_OBJECT_TO_WORLD, shadingNormal);
    bool isHitFromOutside = hitFromOutside(ray.direction, worldShadingNormal);
    if( (isHitFromOutside && ray.ray_type == RayType::RADIANCE_IN_PARTICIPATING_MEDIUM) || 
        (!isHitFromOutside && ray.ray_type == RayType::RADIANCE ))
    {
        //rtPrintf("Ignore int' tHit=%.4f", tHit);
        rtIgnoreIntersection();
    }
}

/*
// Pass the photon along its way through the glass
*/

rtDeclareVariable(PhotonPRD, photonPrd, rtPayload, );

RT_PROGRAM void closestHitPhoton()
{
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    bool isHitFromOutside = hitFromOutside(ray.direction, worldShadingNormal);
    float3 N = isHitFromOutside ? worldShadingNormal : -worldShadingNormal;
    float3 hitPoint = ray.origin + tHit*ray.direction;
   
    
    IndexOfRefractions ior = getIndexOfRefractions(isHitFromOutside, indexOfRefraction);
    float3 refractionDirection;
    bool validRefraction = refract(refractionDirection, ray.direction, N, ior.n2/ior.n1);
    float cosThetaI = -dot(ray.direction, N);
    float cosThetaT = -dot(refractionDirection, N);

    // Find reflection factor using Fresnel equation
    float reflFactor = validRefraction ? reflectionFactor(cosThetaI, cosThetaT, ior.n1, ior.n2) : 1.f;
    float sample = getRandomUniformFloat(&photonPrd.randomState);
    bool isReflected = (sample <= reflFactor);
    float3 newRayDirection;

    if(isReflected)
    {
        newRayDirection = reflect(ray.direction, N);
    }
    else
    {
        newRayDirection = refractionDirection;
    }

    OPTIX_PRINTFID(launchIndex, photonPrd.depth, "Photon hit glass %s (%s) %s P(%.2f %.2f %.2f)\n", isHitFromOutside ? "outside" : "inside",
        willTravelInsideGlass(isHitFromOutside, isReflected)  ? "will travel inside" : "will travel outside", isReflected ? "reflect":"refract", hitPoint.x, hitPoint.y, hitPoint.z);

    photonPrd.depth++;
    if (photonPrd.depth <= MAX_PHOTON_TRACE_DEPTH)
    {
        // If we are going to travel inside the glass object, set ray type to RayType::PHOTON_IN_PARTICIPATING_MEDIUM to
        // prevent interaction with any partcipating medium
        RayType::E rayType = willTravelInsideGlass(isHitFromOutside, isReflected) ? RayType::PHOTON_IN_PARTICIPATING_MEDIUM : RayType::PHOTON;
        Ray newRay(hitPoint, newRayDirection, rayType, 0.0001);
        rtTrace(sceneRootObject, newRay, photonPrd);
    }
}

RT_PROGRAM void anyHitPhoton()
{
    float3 worldShadingNormal = rtTransformNormal(RT_OBJECT_TO_WORLD, shadingNormal);
    bool isHitFromOutside = hitFromOutside(ray.direction, worldShadingNormal);
    if( (isHitFromOutside && ray.ray_type == RayType::PHOTON_IN_PARTICIPATING_MEDIUM) || 
        (!isHitFromOutside && ray.ray_type == RayType::PHOTON ))
    {
        //rtPrintf("Ignore int' tHit=%.4f", tHit);
        rtIgnoreIntersection();
    }
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// Vertex Connection and Merging
#define OPTIX_PRINTF_ENABLED 0
#define OPTIX_PRINTFI_ENABLED 0
#define OPTIX_PRINTFID_ENABLED 0
#define OPTIX_PRINTFC_ENABLED 0
#define OPTIX_PRINTFCID_ENABLED 0

rtDeclareVariable(Camera,     camera, , );
rtDeclareVariable(float2,     pixelSizeFactor, , );
rtDeclareVariable(SubpathPRD, subpathPrd, rtPayload, );
rtDeclareVariable(uint,       lightVertexCountEstimatePass, , );
rtDeclareVariable(uint,       maxPathLen, , );

rtBuffer<LightVertex>  lightVertexBuffer;
rtBuffer<uint>         lightVertexBufferIndexBuffer; // single element buffer with index for lightVertexBuffer
rtBuffer<uint, 2>      lightSubpathLengthBuffer;

rtDeclareVariable(int, lightVertexBufferId, , );            // rtBufferId<LightVertex>
rtDeclareVariable(int, lightVertexBufferIndexBufferId, , ); // rtBufferId<uint>
rtDeclareVariable(int, lightSubpathLengthBufferId, , );     // rtBufferId<uint, 2>
rtDeclareVariable(int, outputBufferId, , );                 // rtBufferId<float3, 2>

#if !VCM_UNIFORM_VERTEX_SAMPLING
rtBuffer<uint, 3>       lightSubpathVertexIndexBuffer;
rtDeclareVariable(int,  lightSubpathVertexIndexBufferId, , ); // rtBufferId<uint, 3>
#else
rtDeclareVariable(float, vertexPickPdf, , );                // used for uniform vertex sampling
#endif


rtDeclareVariable(float, lightSubpathCount, , );
rtDeclareVariable(float, misVcWeightFactor, , ); // 1/etaVCM
rtDeclareVariable(float, misVmWeightFactor, , ); // etaVCM


 // Light subpath program
RT_PROGRAM void vcmClosestHitLight()
{
    float3 worldGeometricNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometricNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;      
    
    bool isHitFromOutside = hitFromOutside(ray.direction, worldGeometricNormal);
    float3 N = isHitFromOutside ? worldGeometricNormal : -worldGeometricNormal;
    IndexOfRefractions ior = getIndexOfRefractions(isHitFromOutside, indexOfRefraction);

    rtBufferId<float3, 2>   _outputBufferId                  = rtBufferId<float3, 2>(outputBufferId);
    rtBufferId<LightVertex> _lightVertexBufferId             = rtBufferId<LightVertex>(lightVertexBufferId);
    rtBufferId<uint>        _lightVertexBufferIndexBufferId  = rtBufferId<uint>(lightVertexBufferIndexBufferId);
#if !VCM_UNIFORM_VERTEX_SAMPLING
    rtBufferId<uint, 3>     _lightSubpathVertexIndexBufferId = rtBufferId<uint, 3>(lightSubpathVertexIndexBufferId);
#endif

    if (isZero(Kr))
        return;
    
    SpecularTransmission transmission(Kr, ior.n1 , ior.n2);
    LightBSDF lightBsdf = LightBSDF(N, -ray.direction);
    lightBsdf.AddBxDF(&transmission);

    lightHit(sceneRootObject, subpathPrd, hitPoint, N, lightBsdf, ray.direction, tHit, maxPathLen,
        lightVertexCountEstimatePass, lightSubpathCount, misVcWeightFactor, misVmWeightFactor,
        camera, pixelSizeFactor,
        _outputBufferId, _lightVertexBufferId, _lightVertexBufferIndexBufferId,
#if !VCM_UNIFORM_VERTEX_SAMPLING
        _lightSubpathVertexIndexBufferId
#else
        &vertexPickPdf
#endif
        );
}


//rtDeclareVariable(uint, vcmNumlightVertexConnections, , );
rtDeclareVariable(float, averageLightSubpathLength, , );
rtDeclareVariable(int,   lightsBufferId, , );                 // rtBufferId<uint, 1>

 // Camra subpath program
RT_PROGRAM void vcmClosestHitCamera()
{
    float3 worldGeometricNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometricNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;      
    
    bool isHitFromOutside = hitFromOutside(ray.direction, worldGeometricNormal);
    float3 N = isHitFromOutside ? worldGeometricNormal : -worldGeometricNormal;
    IndexOfRefractions ior = getIndexOfRefractions(isHitFromOutside, indexOfRefraction);

    rtBufferId<Light>       _lightsBufferId                  = rtBufferId<Light>(lightsBufferId);
    rtBufferId<uint, 2>     _lightSubpathLengthBufferId      = rtBufferId<uint, 2>(lightSubpathLengthBufferId);
    rtBufferId<LightVertex> _lightVertexBufferId             = rtBufferId<LightVertex>(lightVertexBufferId);
    rtBufferId<uint>        _lightVertexBufferIndexBufferId  = rtBufferId<uint>(lightVertexBufferIndexBufferId);
#if !VCM_UNIFORM_VERTEX_SAMPLING
    rtBufferId<uint, 3>     _lightSubpathVertexIndexBufferId = rtBufferId<uint, 3>(lightSubpathVertexIndexBufferId);
#endif

    if (isZero(Kr))
        return;
    
    SpecularTransmission transmission(Kr, ior.n1 , ior.n2);
    CameraBSDF cameraBsdf = CameraBSDF(N, -ray.direction);
    cameraBsdf.AddBxDF(&transmission);

    cameraHit(sceneRootObject, subpathPrd, hitPoint, N, cameraBsdf, ray.direction, tHit, maxPathLen,
         misVcWeightFactor, misVmWeightFactor, 
         _lightsBufferId, _lightSubpathLengthBufferId, _lightVertexBufferId, _lightVertexBufferIndexBufferId,
#if !VCM_UNIFORM_VERTEX_SAMPLING
         _lightSubpathVertexIndexBufferId
#else
         averageLightSubpathLength,
         &vertexPickPdf
#endif
        );
}