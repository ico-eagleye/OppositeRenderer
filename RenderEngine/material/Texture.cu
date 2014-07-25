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
#include "renderer/helpers/random.h"
#include "renderer/helpers/samplers.h"
#include "renderer/helpers/store_photon.h"
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

rtDeclareVariable(rtObject, sceneRootObject, , );
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(RadiancePRD, radiancePrd, rtPayload, );
rtDeclareVariable(PhotonPRD, photonPrd, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, tHit, rtIntersectionDistance, );

rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, ); 
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, ); 
rtDeclareVariable(float3, tangent, attribute tangent, ); 
rtDeclareVariable(float3, bitangent, attribute bitangent, ); 
rtDeclareVariable(float2, textureCoordinate, attribute textureCoordinate, ); 

rtBuffer<Photon, 1> photons;
rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> diffuseSampler;
rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> normalMapSampler;
rtDeclareVariable(unsigned int, hasNormals, , );
rtDeclareVariable(uint, maxPhotonDepositsPerEmitted, , );

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_STOCHASTIC_HASH
rtDeclareVariable(uint3, photonsGridSize, , );
rtDeclareVariable(float3, photonsWorldOrigo, ,);
rtDeclareVariable(float, photonsGridCellSize, ,);
rtDeclareVariable(unsigned int, photonsSize,,);
rtBuffer<unsigned int, 1> photonsHashTableCount;
#endif



__inline__ __device__ float3 getNormalMappedNormal(const float3 & normal, const float3 & tangent, const float3 & bitangent, const float4 & normalMap)
{
    float4 nMap = 2*normalMap - 1;
    float3 N;
    N.x = nMap.x*tangent.x + nMap.y*bitangent.x + nMap.z*normal.x;
    N.y = nMap.x*tangent.y + nMap.y*bitangent.y + nMap.z*normal.y;
    N.z = nMap.x*tangent.z + nMap.y*bitangent.z + nMap.z*normal.z;
    return normalize(N);
}


/*
// Radiance Program
*/
RT_PROGRAM void closestHitRadiance()
{
    float3 worldShadingNormal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shadingNormal));
    float3 hitPoint = ray.origin + tHit*ray.direction;

    float3 normal = worldShadingNormal;
    if(hasNormals)
    {
        float3 worldTangent = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, tangent));
        float3 worldBitangent = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, bitangent));
        normal = getNormalMappedNormal(worldShadingNormal, worldTangent, worldBitangent, 
                        tex2D(normalMapSampler, textureCoordinate.x, textureCoordinate.y));
    }

    radiancePrd.flags |= PRD_HIT_NON_SPECULAR;
    radiancePrd.normal = normal;
    radiancePrd.position = hitPoint;
    radiancePrd.lastTHit = tHit;
    
    if(radiancePrd.flags & PRD_PATH_TRACING)
    {
        radiancePrd.randomNewDirection = sampleUnitHemisphereCos(worldShadingNormal, getRandomUniformFloat2(&radiancePrd.randomState));
    }

    float4 value = tex2D( diffuseSampler, textureCoordinate.x, textureCoordinate.y );
    float3 value3 = make_float3(value.x, value.y, value.z);
    radiancePrd.attenuation *= value3;
}


/*
 Photon Program
*/
RT_PROGRAM void closestHitPhoton()
{
    float3 worldShadingNormal = normalize(rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal));
    float3 normal = worldShadingNormal;
    if(hasNormals)
    {
        float3 worldTangent = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, tangent));
        float3 worldBitangent = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, bitangent));
        normal = getNormalMappedNormal(worldShadingNormal, worldTangent, worldBitangent, 
                        tex2D(normalMapSampler, textureCoordinate.x, textureCoordinate.y));
    }
    float3 hitPoint = ray.origin + tHit*ray.direction;
    float3 newPhotonDirection;

    // Record hit if it has bounced at least once
    if(photonPrd.depth >= 1)
    {
        // vmarz: worldShadingNormal actually is not being stored in photon
        Photon photon (photonPrd.power, hitPoint, ray.direction, worldShadingNormal);
        STORE_PHOTON(photon);
    }

    float4 value = tex2D(diffuseSampler, textureCoordinate.x, textureCoordinate.y);
    float3 value3 = make_float3(value.x, value.y, value.z);
    photonPrd.power *= value3;
#ifdef OPTIX_MATERIAL_DUMP
    for(int i = 0; i<photonPrd.depth;i++) rtPrintf("\t"); 
        rtPrintf("Hit diffuse at P(%.2f %.2f %.2f) t=%.3f\n", hitPoint.x, hitPoint.y, hitPoint.z, tHit);
#endif

    photonPrd.weight *= fmaxf(value3);

    // Use russian roulette sampling from depth X to limit the length of the path

    if( photonPrd.depth >= PHOTON_TRACING_RR_START_DEPTH)
    {
        float probContinue = favgf(value3);
        float probSample = getRandomUniformFloat(&photonPrd.randomState);
        if(probSample >= probContinue )
        {
            return;
        }
        photonPrd.power /= probContinue;
    }

    photonPrd.depth++;
    if(photonPrd.depth >= MAX_PHOTON_TRACE_DEPTH || photonPrd.weight < 0.01)
    {
        return;
    }

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_UNIFORM_GRID || ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_KD_TREE_CPU
    if(photonPrd.numStoredPhotons >= maxPhotonDepositsPerEmitted)
        return;
#endif

    newPhotonDirection = sampleUnitHemisphereCos(worldShadingNormal, getRandomUniformFloat2(&photonPrd.randomState));
    optix::Ray newRay( hitPoint, newPhotonDirection, RayType::PHOTON, 0.01 );
    rtTrace(sceneRootObject, newRay, photonPrd);
}




////////////////////////////////////////////////////////////////////////////////////////////////////
// Vertex Connection and Merging
#define OPTIX_PRINTF_ENABLED 0
#define OPTIX_PRINTFI_ENABLED 0
#define OPTIX_PRINTFID_ENABLED 0

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

    float4 texColor4 = tex2D( diffuseSampler, textureCoordinate.x, textureCoordinate.y );
    float3 texColor = make_float3(texColor4.x, texColor4.y, texColor4.z);
    
    VcmBSDF lightBsdf = VcmBSDF(geometricNormal, -ray.direction, true);
    Lambertian lambertian = Lambertian(texColor);
    lightBsdf.AddBxDF(&lambertian);

    rtBufferId<float3, 2>   _outputBufferId                  = rtBufferId<float3, 2>(outputBufferId);
    rtBufferId<LightVertex> _lightVertexBufferId             = rtBufferId<LightVertex>(lightVertexBufferId);
    rtBufferId<uint>        _lightVertexBufferIndexBufferId  = rtBufferId<uint>(lightVertexBufferIndexBufferId);
    rtBufferId<uint, 2>     _lightSubpathVertexCountBufferId = rtBufferId<uint, 2>(lightSubpathVertexCountBufferId);
#if !VCM_UNIFORM_VERTEX_SAMPLING
    rtBufferId<uint, 3>     _lightSubpathVertexIndexBufferId = rtBufferId<uint, 3>(lightSubpathVertexIndexBufferId);
#endif

    lightHit( sceneRootObject, subpathPrd, hitPoint, worldGeometricNormal, lightBsdf, ray.direction, tHit, maxPathLen,
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

    float4 texColor4 = tex2D( diffuseSampler, textureCoordinate.x, textureCoordinate.y );
    float3 texColor = make_float3(texColor4.x, texColor4.y, texColor4.z);
    
    VcmBSDF cameraBsdf = VcmBSDF(worldGeometricNormal, -ray.direction, false);
    Lambertian lambertian = Lambertian(texColor);
    cameraBsdf.AddBxDF(&lambertian);

    OPTIX_PRINTFID(launchIndex, subpathPrd.depth, "Hit C - texture hit Kd  % 14f % 14f % 14f\n", texColor.x, texColor.y, texColor.z);

    rtBufferId<Light>       _lightsBufferId                  = rtBufferId<Light>(lightsBufferId);
    rtBufferId<LightVertex> _lightVertexBufferId             = rtBufferId<LightVertex>(lightVertexBufferId);
    rtBufferId<uint>        _lightVertexBufferIndexBufferId  = rtBufferId<uint>(lightVertexBufferIndexBufferId);
    rtBufferId<uint, 2>     _lightSubpathVertexCountBufferId = rtBufferId<uint, 2>(lightSubpathVertexCountBufferId);
#if !VCM_UNIFORM_VERTEX_SAMPLING
    rtBufferId<uint, 3>     _lightSubpathVertexIndexBufferId = rtBufferId<uint, 3>(lightSubpathVertexIndexBufferId);
#endif

    cameraHit( sceneRootObject, sceneBoundingSphere, subpathPrd, hitPoint, worldGeometricNormal, cameraBsdf, ray.direction, tHit, maxPathLen,
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