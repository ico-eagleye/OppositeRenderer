/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

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
    OPTIX_DEBUG_PRINT(photonPrd.depth, "Hit Diffuse P(%.2f %.2f %.2f) RT=%d\n", hitPoint.x, hitPoint.y, hitPoint.z, ray.ray_type);
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




rtDeclareVariable(SubpathPRD, lightPrd, rtPayload, );
rtDeclareVariable(uint, lightVertexCountEstimatePass, , );
rtBuffer<uint, 2> lightVertexCountBuffer;
rtBuffer<LightVertex> lightVertexBuffer;
rtBuffer<uint> lightVertexBufferIndexBuffer; // single element buffer with index for lightVertexBuffer

rtDeclareVariable(float, misVcWeightFactor, , ); // 1/etaVCM
rtDeclareVariable(float, misVmWeightFactor, , ); // etaVCM

 // Light subpath program
RT_PROGRAM void closestHitLight()
{
    lightPrd.depth++;	

    // vmarz TODO make sure shading normals used correctly
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;

    //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - inc dir %f %f %f\n", ray.direction.x, ray.direction.y, ray.direction.z);
    //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - point   %f %f %f\n", hitPoint.x, hitPoint.y, hitPoint.z);
    //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - normal  %f %f %f\n", worldShadingNormal.x, worldShadingNormal.y, worldShadingNormal.z);

    // vmarz TODO infinite lights need attitional handling
    float cosThetaIn = dot(worldShadingNormal, -ray.direction);
    //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - cos theta %f \n", hitCosTheta);
    if (cosThetaIn < 0.f) // vmarz TODO check validity
    {
        lightPrd.done = 1;
        return;
    }   

    updateMisTermsOnHit(lightPrd, cosThetaIn, tHit);

    LightVertex lightVertex;
    lightVertex.hitPoint = hitPoint;
    lightVertex.throughput = lightPrd.throughput;
    lightVertex.pathDepth = lightPrd.depth;
    lightVertex.dVCM = lightPrd.dVCM;
    lightVertex.dVC = lightPrd.dVC;
    lightVertex.dVM = lightPrd.dVM;
    // vmarz TODO store material bsdf

    // store path vertex
    if (lightVertexCountEstimatePass) // vmarz: store flag in PRD ?
    {
        lightVertexCountBuffer[launchIndex] = lightPrd.depth;
    }
    else
    {
        uint idx = atomicAdd(&lightVertexBufferIndexBuffer[0], 1u);
        //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - store V %u\n", idx);
        lightVertexBuffer[idx] = lightVertex;
    }

    // vmarz TODO connect to camera
    // vmarz TODO check max path length
    
    // Russian Roulette
    float contProb = luminanceCIE(Kd); // vmarz TODO precompute
    float rrSample = getRandomUniformFloat(&lightPrd.randomState);    
    //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - cnt rr  %f %f \n", contProb, rrSample);
    if (contProb < rrSample)
    {
        lightPrd.done = 1;
        return;
    }

    // next event
    float3 bsdfFactor = Kd * M_1_PIf;
    float bsdfDirPdfW;
    float cosThetaOut;
    float2 bsdfSample = getRandomUniformFloat2(&lightPrd.randomState);
    lightPrd.direction = sampleUnitHemisphereCos(worldShadingNormal, bsdfSample, &bsdfDirPdfW, &cosThetaOut);
    //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - new dir %f %f %f\n", lightPrd.direction.x, lightPrd.direction.y, lightPrd.direction.z);

    float bsdfRevPdfW = cosThetaIn * M_1_PIf;
    bsdfDirPdfW *= contProb;
    bsdfRevPdfW *= contProb;
    updateMisTermsOnScatter(lightPrd, cosThetaOut, bsdfDirPdfW, bsdfRevPdfW, misVcWeightFactor, misVmWeightFactor);

    // f * cosTheta / f_pdf
    lightPrd.throughput *= bsdfFactor * (cosThetaOut / bsdfDirPdfW);
    lightPrd.origin = hitPoint;
    //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - new org %f %f %f\n", lightPrd.origin.x, lightPrd.origin.y, lightPrd.origin.z);
}


rtDeclareVariable(uint, vcmNumlightVertexConnections, , );

 // Camra subpath program
RT_PROGRAM void vcmClosestHitCamera()
{
    lightPrd.depth++;	

    // vmarz TODO make sure shading normals used correctly
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;

    //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - inc dir %f %f %f\n", ray.direction.x, ray.direction.y, ray.direction.z);
    //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - point   %f %f %f\n", hitPoint.x, hitPoint.y, hitPoint.z);
    //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - normal  %f %f %f\n", worldShadingNormal.x, worldShadingNormal.y, worldShadingNormal.z);

    // vmarz TODO infinite lights need attitional handling
    float cosThetaIn = dot(worldShadingNormal, -ray.direction);
    //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - cos theta %f \n", hitCosTheta);
    if (cosThetaIn < 0.f) // vmarz TODO check validity
    {
        lightPrd.done = 1;
        return;
    }   

    updateMisTermsOnHit(lightPrd, cosThetaIn, tHit);

    // Connect to ligth vertices
    //for (int i = 1; i<vcmNumlightVertexConnections; i++)
    //{
    //    uint numLightVertices = lightVertexBufferIndexBuffer[0];
    //    uint vertIdx = numLightVertices * getRandomUniformFloat(&lightPrd.randomState);
    //    float vertexPickPdf = float(vcmNumlightVertexConnections) / numLightVertices;
    //    LightVertex lightVertex = lightVertexBuffer[vertIdx];
    //}

    // vmarz TODO check max path length
    // Russian Roulette
    float contProb = luminanceCIE(Kd); // vmarz TODO precompute
    float rrSample = getRandomUniformFloat(&lightPrd.randomState);    
    //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - cnt rr  %f %f \n", contProb, rrSample);
    if (contProb < rrSample)
    {
        lightPrd.done = 1;
        return;
    }

    // next event
    float3 bsdfFactor = Kd * M_1_PIf;
    float bsdfDirPdfW;
    float cosThetaOut;
    float2 bsdfSample = getRandomUniformFloat2(&lightPrd.randomState);
    lightPrd.direction = sampleUnitHemisphereCos(worldShadingNormal, bsdfSample, &bsdfDirPdfW, &cosThetaOut);
    //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - new dir %f %f %f\n", lightPrd.direction.x, lightPrd.direction.y, lightPrd.direction.z);

    float bsdfRevPdfW = cosThetaIn * M_1_PIf;
    bsdfDirPdfW *= contProb;
    bsdfRevPdfW *= contProb;
    updateMisTermsOnScatter(lightPrd, cosThetaOut, bsdfDirPdfW, bsdfRevPdfW, misVcWeightFactor, misVmWeightFactor);

    // f * cosTheta / f_pdf
    lightPrd.throughput *= bsdfFactor * (cosThetaOut / bsdfDirPdfW);
    lightPrd.origin = hitPoint;
    //OPTIX_DEBUG_PRINT(lightPrd.depth, "Hit - new org %f %f %f\n", lightPrd.origin.x, lightPrd.origin.y, lightPrd.origin.z);
}


optix::float3 __inline __device__ connectVertices(LightVertex aVertex, SubpathPRD & aCameraPrd, optix::float3 aCameraHitpoint)
{
    // check occlusion
}