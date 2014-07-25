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
#include "renderer/RayType.h"
#include "renderer/RadiancePRD.h"
#include "renderer/ppm/PhotonPRD.h"
#include "renderer/ppm/Photon.h"
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

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(PhotonPRD, photonPrd, rtPayload, );
rtDeclareVariable(RadiancePRD, radiancePrd, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, tHit, rtIntersectionDistance, );

rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, ); 
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, ); 

rtDeclareVariable(rtObject, sceneRootObject, , );
rtDeclareVariable(float3, Kr, , );


RT_PROGRAM void closestHitRadiance()
{
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;
    radiancePrd.depth++;
    if(radiancePrd.depth <= MAX_RADIANCE_TRACE_DEPTH)
    {
        radiancePrd.attenuation *= Kr;
        float3 newRayDirection = reflect(ray.direction, worldShadingNormal);
        Ray newRay ( hitPoint, newRayDirection, RayType::RADIANCE, 0.0001, RT_DEFAULT_MAX );
        rtTrace( sceneRootObject, newRay, radiancePrd );
    }
    radiancePrd.lastTHit = tHit;
}

RT_PROGRAM void closestHitPhoton()
{
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;
    photonPrd.depth++;
    if (photonPrd.depth <= MAX_PHOTON_TRACE_DEPTH)
    {
        photonPrd.power *= Kr;
        float3 newPhotonDirection = reflect(ray.direction, worldShadingNormal);
        Ray newPhoton (hitPoint, newPhotonDirection, RayType::PHOTON, 0.0001 );
        rtTrace(sceneRootObject, newPhoton, photonPrd);
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
rtBuffer<uint, 2>      lightSubpathVertexCountBuffer;

rtDeclareVariable(int, lightVertexBufferId, , );            // rtBufferId<LightVertex>
rtDeclareVariable(int, lightVertexBufferIndexBufferId, , ); // rtBufferId<uint>
rtDeclareVariable(int, lightSubpathVertexCountBufferId, , );// rtBufferId<uint, 2>
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
    
    rtBufferId<float3, 2>   _outputBufferId                  = rtBufferId<float3, 2>(outputBufferId);
    rtBufferId<LightVertex> _lightVertexBufferId             = rtBufferId<LightVertex>(lightVertexBufferId);
    rtBufferId<uint>        _lightVertexBufferIndexBufferId  = rtBufferId<uint>(lightVertexBufferIndexBufferId);
    rtBufferId<uint, 2>     _lightSubpathVertexCountBufferId = rtBufferId<uint, 2>(lightSubpathVertexCountBufferId);
#if !VCM_UNIFORM_VERTEX_SAMPLING
    rtBufferId<uint, 3>     _lightSubpathVertexIndexBufferId = rtBufferId<uint, 3>(lightSubpathVertexIndexBufferId);
#endif

    if (isZero(Kr))
        return;
    
    VcmBSDF lightBsdf = VcmBSDF(worldGeometricNormal, -ray.direction, true);
    FresnelNoOp fresnelNoOp;
    SpecularReflection reflection(Kr, &fresnelNoOp );
    lightBsdf.AddBxDF(&reflection);

    lightHit(sceneRootObject, subpathPrd, hitPoint, worldGeometricNormal, lightBsdf, ray.direction, tHit, maxPathLen,
        lightVertexCountEstimatePass, lightSubpathCount, misVcWeightFactor, misVmWeightFactor,
        camera, pixelSizeFactor,
        _outputBufferId, _lightVertexBufferId, _lightVertexBufferIndexBufferId, _lightSubpathVertexCountBufferId, 
#if !VCM_UNIFORM_VERTEX_SAMPLING
        _lightSubpathVertexIndexBufferId
#else
        &vertexPickPdf
#endif
        );
}


//rtDeclareVariable(uint, vcmNumlightVertexConnections, , );
rtDeclareVariable(Sphere, sceneBoundingSphere, , );
rtDeclareVariable(float,  averageLightSubpathLength, , );
rtDeclareVariable(int,    lightsBufferId, , );                 // rtBufferId<uint, 1>

 // Camra subpath program
RT_PROGRAM void vcmClosestHitCamera()
{
    float3 worldGeometricNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometricNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;

    rtBufferId<Light>       _lightsBufferId                  = rtBufferId<Light>(lightsBufferId);
    rtBufferId<LightVertex> _lightVertexBufferId             = rtBufferId<LightVertex>(lightVertexBufferId);
    rtBufferId<uint>        _lightVertexBufferIndexBufferId  = rtBufferId<uint>(lightVertexBufferIndexBufferId);
    rtBufferId<uint, 2>     _lightSubpathVertexCountBufferId = rtBufferId<uint, 2>(lightSubpathVertexCountBufferId);
#if !VCM_UNIFORM_VERTEX_SAMPLING
    rtBufferId<uint, 3>     _lightSubpathVertexIndexBufferId = rtBufferId<uint, 3>(lightSubpathVertexIndexBufferId);
#endif

    if (isZero(Kr))
        return;
    
    VcmBSDF cameraBsdf = VcmBSDF(worldGeometricNormal, -ray.direction, false);
    FresnelNoOp fresnelNoOp;
    SpecularReflection reflection(Kr, &fresnelNoOp );
    cameraBsdf.AddBxDF(&reflection);

    cameraHit(sceneRootObject, sceneBoundingSphere, subpathPrd, hitPoint, worldGeometricNormal, cameraBsdf, ray.direction, tHit, maxPathLen,
         misVcWeightFactor, misVmWeightFactor, 
         _lightsBufferId, _lightVertexBufferId, _lightVertexBufferIndexBufferId, _lightSubpathVertexCountBufferId, 
#if !VCM_UNIFORM_VERTEX_SAMPLING
         _lightSubpathVertexIndexBufferId
#else
         averageLightSubpathLength,
         &vertexPickPdf
#endif
        );
}