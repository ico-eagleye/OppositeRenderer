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
#include "material/VcmBsdfData.h"
#include "material/VcmBsdfEvalProgram.h"
#include "material/BxDF.h"
#include "material/BSDF.h"

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




rtDeclareVariable(SubpathPRD, subpathPrd, rtPayload, );
rtDeclareVariable(uint, lightVertexCountEstimatePass, , );
rtBuffer<uint, 2> lightVertexCountBuffer;
rtBuffer<LightVertex> lightVertexBuffer;
rtBuffer<uint> lightVertexBufferIndexBuffer; // single element buffer with index for lightVertexBuffer

rtDeclareVariable(float, misVcWeightFactor, , ); // 1/etaVCM
rtDeclareVariable(float, misVmWeightFactor, , ); // etaVCM


__device__ __inline__ VcmBSDF getVcmBSDF(float3 & aNormal, float3 & aHitDir)
{
    VcmBSDF bsdf = VcmBSDF(aNormal, aHitDir);
    //Lambertian * lambertian = reinterpret_cast<Lambertian *>(bsdf.bxdfAt(0));
    //*lambertian = Lambertian(Kd);
    return bsdf;
}


 // Light subpath program
RT_PROGRAM void closestHitLight()
{
    subpathPrd.depth++;	

    // vmarz TODO make sure shading normals used correctly
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;

    //OPTIX_DEBUG_PRINT(subpathPrd.depth, "Hit - inc dir %f %f %f\n", ray.direction.x, ray.direction.y, ray.direction.z);
    //OPTIX_DEBUG_PRINT(subpathPrd.depth, "Hit - point   %f %f %f\n", hitPoint.x, hitPoint.y, hitPoint.z);
    //OPTIX_DEBUG_PRINT(subpathPrd.depth, "Hit - normal  %f %f %f\n", worldShadingNormal.x, worldShadingNormal.y, worldShadingNormal.z);

    // vmarz TODO infinite lights need attitional handling
    float cosThetaIn = dot(worldShadingNormal, -ray.direction);
    //OPTIX_DEBUG_PRINT(subpathPrd.depth, "Hit - cos theta %f \n", hitCosTheta);
    if (cosThetaIn < 0.f) // vmarz TODO check validity
    {
        subpathPrd.done = 1;
        return;
    }   

    updateMisTermsOnHit(subpathPrd, cosThetaIn, tHit);

    LightVertex lightVertex;
    lightVertex.hitPoint = hitPoint;
    lightVertex.throughput = subpathPrd.throughput;
    lightVertex.pathDepth = subpathPrd.depth;
    lightVertex.dVCM = subpathPrd.dVCM;
    lightVertex.dVC = subpathPrd.dVC;
    lightVertex.dVM = subpathPrd.dVM;
    //lightVertex.bsdf = getVcmBSDF(shadingNormal, ray.direction);
    //lightVertex.bsdfData.material = VcmMeterial::DIFFUSE;
    //lightVertex.bsdfData.bsdfDiffuse.Kd = Kd;

    // store path vertex
    if (lightVertexCountEstimatePass) // vmarz: store flag in PRD ?
    {
        lightVertexCountBuffer[launchIndex] = subpathPrd.depth;
    }
    else
    {
        uint idx = atomicAdd(&lightVertexBufferIndexBuffer[0], 1u);
        //OPTIX_DEBUG_PRINT(subpathPrd.depth, "Hit - store V %u\n", idx);
        lightVertexBuffer[idx] = lightVertex;
    }

    // vmarz TODO connect to camera
    // vmarz TODO check max path length
    
    // Russian Roulette
    float contProb = luminanceCIE(Kd); // vmarz TODO precompute
    float rrSample = getRandomUniformFloat(&subpathPrd.randomState);    
    //OPTIX_DEBUG_PRINT(subpathPrd.depth, "Hit - cnt rr  %f %f \n", contProb, rrSample);
    if (contProb < rrSample)
    {
        subpathPrd.done = 1;
        return;
    }

    // next event
    float3 bsdfFactor = Kd * M_1_PIf;
    float bsdfDirPdfW;
    float cosThetaOut;
    float2 bsdfSample = getRandomUniformFloat2(&subpathPrd.randomState);
    subpathPrd.direction = sampleUnitHemisphereCos(worldShadingNormal, bsdfSample, &bsdfDirPdfW, &cosThetaOut);
    //OPTIX_DEBUG_PRINT(subpathPrd.depth, "Hit - new dir %f %f %f\n", subpathPrd.direction.x, subpathPrd.direction.y, subpathPrd.direction.z);

    float bsdfRevPdfW = cosThetaIn * M_1_PIf;
    bsdfDirPdfW *= contProb;
    bsdfRevPdfW *= contProb;
    updateMisTermsOnScatter(subpathPrd, cosThetaOut, bsdfDirPdfW, bsdfRevPdfW, misVcWeightFactor, misVmWeightFactor);

    // f * cosTheta / f_pdf
    subpathPrd.throughput *= bsdfFactor * (cosThetaOut / bsdfDirPdfW);
    subpathPrd.origin = hitPoint;
    //OPTIX_DEBUG_PRINT(subpathPrd.depth, "Hit - new org %f %f %f\n", subpathPrd.origin.x, subpathPrd.origin.y, subpathPrd.origin.z);
}



__inline
__device__ int isOccluded(optix::float3 point, optix::float3 direction, float tMax)
{
    ShadowPRD shadowPrd;
    shadowPrd.attenuation = 1.0f;
    Ray occlusionRay(point, direction, RayType::SHADOW, EPS_RAY, tMax - 2.f*EPS_RAY);
    rtTrace(sceneRootObject, occlusionRay, shadowPrd);
    return shadowPrd.attenuation != 0.f;
}


// Connects vertices and accumulates path contribution in aCameraPrd.color
__device__ void connectVertices(LightVertex & aVertex, VcmBSDF & aCameraBsdf, SubpathPRD & aCameraPrd, optix::float3 & aCameraHitpoint)
{
    //rtPrintf("%d %d - d %d - conn  %f %f %f and %f %f %f\n", aCameraPrd.launchIndex.x, aCameraPrd.launchIndex.y,
    //    aCameraPrd.depth, aCameraHitpoint.x, aCameraHitpoint.y, aCameraHitpoint.z,
    //    aVertex.hitPoint.x, aVertex.hitPoint.y, aVertex.hitPoint.z);
    // check occlusion

    // Get connection
    float3 direction = aVertex.hitPoint - aCameraHitpoint;
    float dist2      = dot(direction, direction);
    float distance   = sqrt(dist2);
    direction       /= distance;

    // Evaluate BSDF at camera vertex
    float directPdfW, reversePdfW;
    const float3 cameraBsdfFactor = aCameraBsdf.vcmF(direction, &directPdfW, &reversePdfW);
    if (isZero(cameraBsdfFactor))
        return;

    if (isOccluded(aCameraHitpoint, direction, distance))
        return;
    
    //Lambertian * lambertian = reinterpret_cast<Lambertian *>(aVertex.bsdf.bxdfAt(0));
    //float3 kd = lambertian->rho(1, NULL, NULL);
    //rtPrintf("%d %d Unoccluded vetext at %f %f %f dirFix %f %f %f Kd %f %f %f\n",
    //    aCameraPrd.launchIndex.x, aCameraPrd.launchIndex.y,
    //    aVertex.hitPoint.x, aVertex.hitPoint.y, aVertex.hitPoint.z,
    //    aVertex.bsdf.localDirFix().x, aVertex.bsdf.localDirFix().y, aVertex.bsdf.localDirFix().z,
    //    kd.x, kd.y, kd.z);

    float cosLight; // cos of incident vector at light vertex from camera vertex
}




rtDeclareVariable(uint, vcmNumlightVertexConnections, , );

 // Camra subpath program
RT_PROGRAM void vcmClosestHitCamera()
{
    //OPTIX_DEBUG_PRINT(subpathPrd.depth, "CamHit\n");
    subpathPrd.depth++;	

    // vmarz TODO make sure shading normals used correctly
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;

    //OPTIX_DEBUG_PRINT(subpathPrd.depth, "Hit - inc dir %f %f %f\n", ray.direction.x, ray.direction.y, ray.direction.z);
    //OPTIX_DEBUG_PRINT(subpathPrd.depth, "Hit - point   %f %f %f\n", hitPoint.x, hitPoint.y, hitPoint.z);
    //OPTIX_DEBUG_PRINT(subpathPrd.depth, "Hit - normal  %f %f %f\n", worldShadingNormal.x, worldShadingNormal.y, worldShadingNormal.z);

    // vmarz TODO infinite lights need attitional handling
    float cosThetaIn = dot(worldShadingNormal, -ray.direction);
    //OPTIX_DEBUG_PRINT(subpathPrd.depth, "Hit - cos theta %f \n", hitCosTheta);
    if (cosThetaIn < 0.f) // vmarz TODO check validity
    {
        subpathPrd.done = 1;
        return;
    }   

    updateMisTermsOnHit(subpathPrd, cosThetaIn, tHit);
    VcmBSDF cameraBsdf = VcmBSDF(worldShadingNormal, ray.direction);

    // TODO connect to light source

    // Connect to ligth vertices
    for (int i = 1; i < vcmNumlightVertexConnections; i++)
    {
        uint numLightVertices = lightVertexBufferIndexBuffer[0];
        uint vertIdx = numLightVertices * getRandomUniformFloat(&subpathPrd.randomState);
        float vertexPickPdf = float(vcmNumlightVertexConnections) / numLightVertices;
        LightVertex lightVertex = lightVertexBuffer[vertIdx];
        connectVertices(lightVertex, cameraBsdf, subpathPrd, hitPoint);
    }
    
    // vmarz TODO check max path length
    // Russian Roulette
    float contProb = luminanceCIE(Kd); // vmarz TODO precompute
    float rrSample = getRandomUniformFloat(&subpathPrd.randomState);    
    //OPTIX_DEBUG_PRINT(subpathPrd.depth, "Hit - cnt rr  %f %f \n", contProb, rrSample);
    if (contProb < rrSample)
    {
        subpathPrd.done = 1;
        return;
    }

    // next event
    float3 bsdfFactor = Kd * M_1_PIf;
    float bsdfDirPdfW;
    float cosThetaOut;
    float2 bsdfSample = getRandomUniformFloat2(&subpathPrd.randomState);
    subpathPrd.direction = sampleUnitHemisphereCos(worldShadingNormal, bsdfSample, &bsdfDirPdfW, &cosThetaOut);
    //OPTIX_DEBUG_PRINT(subpathPrd.depth, "Hit - new dir %f %f %f\n", subpathPrd.direction.x, subpathPrd.direction.y, subpathPrd.direction.z);

    float bsdfRevPdfW = cosThetaIn * M_1_PIf;
    bsdfDirPdfW *= contProb;
    bsdfRevPdfW *= contProb;
    updateMisTermsOnScatter(subpathPrd, cosThetaOut, bsdfDirPdfW, bsdfRevPdfW, misVcWeightFactor, misVmWeightFactor);

    // f * cosTheta / f_pdf
    subpathPrd.throughput *= bsdfFactor * (cosThetaOut / bsdfDirPdfW);
    subpathPrd.origin = hitPoint;
    //OPTIX_DEBUG_PRINT(subpathPrd.depth, "Hit - new org %f %f %f\n", subpathPrd.origin.x, subpathPrd.origin.y, subpathPrd.origin.z);
}