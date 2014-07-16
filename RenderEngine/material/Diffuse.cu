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
#include "renderer/RayType.h"
#include "renderer/RadiancePRD.h"
#include "renderer/ppm/PhotonPRD.h"
#include "renderer/ppm/Photon.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/helpers.h"
#include "renderer/helpers/samplers.h"
#include "renderer/helpers/store_photon.h"
#include "renderer/vcm/SubpathPRD.h"
#include "renderer/vcm/LightVertex.h"
#include "renderer/vcm/vcm.h"
#include "renderer/vcm/mis.h"
#include "renderer/vcm/config_vcm.h"
#include "renderer/BxDF.h"
#include "renderer/BSDF.h"

#define OPTIX_PRINTF_ENABLED 1
#define OPTIX_PRINTFI_ENABLED 1
#define OPTIX_PRINTFID_ENABLED 1

using namespace optix;

rtDeclareVariable(uint2, launchDim, rtLaunchDim, );
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(RadiancePRD, radiancePrd, rtPayload, );
rtDeclareVariable(PhotonPRD, photonPrd, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, tHit, rtIntersectionDistance, );

rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, ); 
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, ); 

rtBuffer<Photon, 1> photons;
rtBuffer<Hitpoint, 2> raytracePassOutputBuffer;
rtDeclareVariable(rtObject, sceneRootObject, , );
rtDeclareVariable(uint, maxPhotonDepositsPerEmitted, , );
rtDeclareVariable(float3, Kd, , );

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_STOCHASTIC_HASH
rtDeclareVariable(uint3, photonsGridSize, , );
rtDeclareVariable(float3, photonsWorldOrigo, ,);
rtDeclareVariable(float, photonsGridCellSize, ,);
rtDeclareVariable(unsigned int, photonsSize,,);
rtBuffer<unsigned int, 1> photonsHashTableCount;
#endif


/*
// Radiance Program
*/
RT_PROGRAM void closestHitRadiance()
{
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;

    radiancePrd.flags |= PRD_HIT_NON_SPECULAR;
    radiancePrd.attenuation *= Kd;
    radiancePrd.normal = worldShadingNormal;
    radiancePrd.position = hitPoint;
    radiancePrd.lastTHit = tHit;
    radiancePrd.depth++; // vmarz: using for debugging (was already defined in struct)
    if(radiancePrd.flags & PRD_PATH_TRACING)
    {
        float2 sample = getRandomUniformFloat2(&radiancePrd.randomState);
        radiancePrd.randomNewDirection = sampleUnitHemisphereCos(worldShadingNormal, sample);
    }
}

/*
// Photon Program
*/
RT_PROGRAM void closestHitPhoton()
{
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;
    float3 newPhotonDirection;

    if(photonPrd.depth >= 1 && photonPrd.numStoredPhotons < maxPhotonDepositsPerEmitted)
    {
        Photon photon (photonPrd.power, hitPoint, ray.direction, worldShadingNormal);
        STORE_PHOTON(photon);
    }

    photonPrd.power *= Kd;
    OPTIX_PRINTFID(launchIndex, photonPrd.depth, "Hit Diffuse P(%.2f %.2f %.2f) RT=%d\n", hitPoint.x, hitPoint.y, hitPoint.z, ray.ray_type);
    photonPrd.weight *= fmaxf(Kd);

    // Use russian roulette sampling from depth X to limit the length of the path

    if( photonPrd.depth >= PHOTON_TRACING_RR_START_DEPTH)
    {
        float probContinue = favgf(Kd);
        float probSample = getRandomUniformFloat(&photonPrd.randomState);
        if(probSample >= probContinue )
        {
            return;
        }
        photonPrd.power /= probContinue;
    }

    photonPrd.depth++;
    if(photonPrd.depth >= MAX_PHOTON_TRACE_DEPTH || photonPrd.weight < 0.001)
    {
        return;
    }

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_UNIFORM_GRID || ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_KD_TREE_CPU
    if(photonPrd.numStoredPhotons >= maxPhotonDepositsPerEmitted)
        return;
#endif

    newPhotonDirection = sampleUnitHemisphereCos(worldShadingNormal, getRandomUniformFloat2(&photonPrd.randomState));
    optix::Ray newRay( hitPoint, newPhotonDirection, RayType::PHOTON, 0.0001 );
    rtTrace(sceneRootObject, newRay, photonPrd);
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// Vertex Connection and Merging
#define OPTIX_PRINTF_ENABLED 1
#define OPTIX_PRINTFI_ENABLED 1
#define OPTIX_PRINTFID_ENABLED 1

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
    // vmarz TODO make sure shading normals used correctly
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;      
    
    rtBufferId<float3, 2>   _outputBufferId                  = rtBufferId<float3, 2>(outputBufferId);
    rtBufferId<LightVertex> _lightVertexBufferId             = rtBufferId<LightVertex>(lightVertexBufferId);
    rtBufferId<uint>        _lightVertexBufferIndexBufferId  = rtBufferId<uint>(lightVertexBufferIndexBufferId);
#if !VCM_UNIFORM_VERTEX_SAMPLING
    rtBufferId<uint, 3>     _lightSubpathVertexIndexBufferId = rtBufferId<uint, 3>(lightSubpathVertexIndexBufferId);
#endif

    Lambertian lambertian = Lambertian(Kd);
    LightBSDF lightBsdf = LightBSDF(worldShadingNormal, -ray.direction);
    lightBsdf.AddBxDF(&lambertian);

    lightHit(sceneRootObject, subpathPrd, hitPoint, worldShadingNormal, lightBsdf, ray.direction, tHit, maxPathLen,
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
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;

    Lambertian lambertian = Lambertian(Kd);
    CameraBSDF cameraBsdf = CameraBSDF(worldShadingNormal, -ray.direction);
    cameraBsdf.AddBxDF(&lambertian);

    rtBufferId<Light>       _lightsBufferId                  = rtBufferId<Light>(lightsBufferId);
    rtBufferId<uint, 2>     _lightSubpathLengthBufferId      = rtBufferId<uint, 2>(lightSubpathLengthBufferId);
    rtBufferId<LightVertex> _lightVertexBufferId             = rtBufferId<LightVertex>(lightVertexBufferId);
    rtBufferId<uint>        _lightVertexBufferIndexBufferId  = rtBufferId<uint>(lightVertexBufferIndexBufferId);
#if !VCM_UNIFORM_VERTEX_SAMPLING
    rtBufferId<uint, 3>     _lightSubpathVertexIndexBufferId = rtBufferId<uint, 3>(lightSubpathVertexIndexBufferId);
#endif

    cameraHit(sceneRootObject, subpathPrd, hitPoint, worldShadingNormal, cameraBsdf, ray.direction, tHit, maxPathLen,
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