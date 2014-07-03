/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#define OPTIX_PRINTFID_DISABLE
#define OPTIX_PRINTFI_DISABLE
#define OPTIX_PRINTFIALL_DISABLE

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
    OPTIX_PRINTFI(photonPrd.depth, "Hit Diffuse P(%.2f %.2f %.2f) RT=%d\n", hitPoint.x, hitPoint.y, hitPoint.z, ray.ray_type);
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

//__noinline__ // seems to cause above error
// "_rtContextCompile" caught exception: Assertion failed: "insn->isMove() || insn->isLoad() || insn->isAdd()", [5639172]
__device__ __inline__ void setVcmBSDF(VcmBSDF &bsdf, float3 & aWorldNormal, float3 & aWorldHitDir)
{
    Lambertian lambertian = Lambertian(Kd);
    //OPTIX_PRINTF("setVcmBSDF - Lambertian._reflectance %f %f %f addr 0x%X\n", 
    //    lambertian._reflectance.x, lambertian._reflectance.y, lambertian._reflectance.z, 
    //    (optix::optix_size_t)&lambertian._reflectance);
    bsdf = VcmBSDF(aWorldNormal, aWorldHitDir);
    bsdf.AddBxDF(&lambertian);
}


 // Light subpath program
RT_PROGRAM void closestHitLight()
{
    subpathPrd.depth++;	

    // vmarz TODO make sure shading normals used correctly
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;

    OPTIX_PRINTFI(subpathPrd.depth, "Hit L - incident dir W  % 14f % 14f % 14f\n", ray.direction.x, ray.direction.y, ray.direction.z);
    OPTIX_PRINTFI(subpathPrd.depth, "Hit L - hitPoint        % 14f % 14f % 14f\n", hitPoint.x, hitPoint.y, hitPoint.z);
    OPTIX_PRINTFI(subpathPrd.depth, "Hit L - normal W        % 14f % 14f % 14f\n", worldShadingNormal.x, worldShadingNormal.y, worldShadingNormal.z);

    // vmarz TODO infinite lights need attitional handling
    float cosThetaIn = dot(worldShadingNormal, -ray.direction);
    OPTIX_PRINTFI(subpathPrd.depth, "Hit L - cos theta %f \n", cosThetaIn);
    if (cosThetaIn < EPS_COSINE) // reject if cos too low
    {
        subpathPrd.done = 1;
        return;
    }   

    updateMisTermsOnHit(subpathPrd, cosThetaIn, tHit);

    LightVertex lightVertex;
    lightVertex.launchIndex = subpathPrd.launchIndex;
    lightVertex.hitPoint = hitPoint;
    lightVertex.throughput = subpathPrd.throughput;
    lightVertex.pathDepth = subpathPrd.depth;
    lightVertex.dVCM = subpathPrd.dVCM;
    lightVertex.dVC = subpathPrd.dVC;
    lightVertex.dVM = subpathPrd.dVM;
    setVcmBSDF(lightVertex.bsdf, worldShadingNormal, -ray.direction);

    DifferentialGeometry dg = lightVertex.bsdf.differentialGeometry();
    OPTIX_PRINTFI(subpathPrd.depth, "Hit L - frame vectors b % 14f % 14f % 14f\n", dg.bitangent.x, dg.bitangent.y, dg.bitangent.z);
    OPTIX_PRINTFI(subpathPrd.depth, "Hit L -               t % 14f % 14f % 14f\n", dg.tangent.x, dg.tangent.y, dg.tangent.z);
    OPTIX_PRINTFI(subpathPrd.depth, "Hit L -               n % 14f % 14f % 14f\n", dg.normal.x, dg.normal.y, dg.normal.z);
    float3 dirFix = lightVertex.bsdf.localDirFix();
    OPTIX_PRINTFI(subpathPrd.depth, "Hit L - dir fix local   % 14f % 14f % 14f\n", dirFix.x, dirFix.y, dirFix.z);

    // store path vertex
    if (lightVertexCountEstimatePass) // vmarz: store flag in PRD ?
    {
        lightVertexCountBuffer[launchIndex] = subpathPrd.depth;
    }
    else
    {
        uint idx = atomicAdd(&lightVertexBufferIndexBuffer[0], 1u);
        OPTIX_PRINTFI(subpathPrd.depth, "Hit L - Vert.throuhput  % 14f % 14f % 14f\n", 
          lightVertex.throughput.x, lightVertex.throughput.y, lightVertex.throughput.z);
        lightVertexBuffer[idx] = lightVertex;
    }

    // vmarz TODO connect to camera
    // vmarz TODO check max path length
    
    // Russian Roulette
    float contProb =  lightVertex.bsdf.continuationProb(); //luminanceCIE(Kd); // vmarz TODO precompute
    float rrSample = getRandomUniformFloat(&subpathPrd.randomState);    
    OPTIX_PRINTFI(subpathPrd.depth, "Hit C - continue sample % 14f             RR % 14f \n", contProb, rrSample);
    if (contProb < rrSample)
    {
        subpathPrd.done = 1;
        return;
    }

    // TODO use BSDF class
    // next event
    float3 bsdfFactor = Kd * M_1_PIf;
    float bsdfDirPdfW;
    float cosThetaOut;
    float2 bsdfSample = getRandomUniformFloat2(&subpathPrd.randomState);
    subpathPrd.direction = sampleUnitHemisphereCos(worldShadingNormal, bsdfSample, &bsdfDirPdfW, &cosThetaOut);
    OPTIX_PRINTFI(subpathPrd.depth, "Hit L - new dir World   % 14f % 14f % 14f\n",
      subpathPrd.direction.x, subpathPrd.direction.y, subpathPrd.direction.z);

    float bsdfRevPdfW = cosThetaIn * M_1_PIf;
    bsdfDirPdfW *= contProb;
    bsdfRevPdfW *= contProb;
    updateMisTermsOnScatter(subpathPrd, cosThetaOut, bsdfDirPdfW, bsdfRevPdfW, misVcWeightFactor, misVmWeightFactor);

    // f * cosTheta / f_pdf
    subpathPrd.throughput *= bsdfFactor * (cosThetaOut / bsdfDirPdfW);
    OPTIX_PRINTFI(subpathPrd.depth, "Hit L - cosThetaOut     % 14f    bsdfDirPdfW % 14f\n", 
      cosThetaOut, bsdfDirPdfW);
    OPTIX_PRINTFI(subpathPrd.depth, "Hit L - prd.througput   % 14f % 14f % 14f\n", 
      subpathPrd.throughput.x, subpathPrd.throughput.y, subpathPrd.throughput.z);

    subpathPrd.origin = hitPoint;
    OPTIX_PRINTFI(subpathPrd.depth, "Hit L - new origin      % 14f % 14f % 14f\n\n", 
      subpathPrd.origin.x, subpathPrd.origin.y, subpathPrd.origin.z);
}



__inline
__device__ int isOccluded(optix::float3 point, optix::float3 direction, float tMax)
{
    ShadowPRD shadowPrd;
    shadowPrd.attenuation = 1.0f;
    Ray occlusionRay(point, direction, RayType::SHADOW, EPS_RAY, tMax - 2.f*EPS_RAY);
    rtTrace(sceneRootObject, occlusionRay, shadowPrd);
    return shadowPrd.attenuation == 0.f;
}

#define OPTIX_PRINTF_FUN printf

//               //
// Connects vertices and accumulates path contribution in aCameraPrd.color
__device__ void connectVertices(LightVertex & alightVertex, float alightVertexPickPdf, VcmBSDF & aCameraBsdf, SubpathPRD & aCameraPrd,
                                optix::float3 & aCameraHitpoint)
{
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  - cameraHitPoint  % 14f % 14f % 14f\n",
        aCameraHitpoint.x, aCameraHitpoint.y, aCameraHitpoint.z);
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  - --> vertex      % 14f % 14f % 14f             id %3d %3d \n",
        alightVertex.hitPoint.x, alightVertex.hitPoint.y, alightVertex.hitPoint.z, 
        alightVertex.launchIndex.x, alightVertex.launchIndex.y);

    // Get connection
    float3 direction = alightVertex.hitPoint - aCameraHitpoint;
    float dist2      = dot(direction, direction);
    float distance   = sqrt(dist2);
    direction       /= distance;
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  -            dir  % 14f % 14f % 14f           dist % 14f\n",
      direction.x, direction.y, direction.z, distance);

    // Evaluate BSDF at camera vertex
    float cameraCosTheta, cameraBsdfDirPdfW, cameraBsdfRevPdfW;
    const float3 cameraBsdfFactor = aCameraBsdf.vcmF(direction, cameraCosTheta, &cameraBsdfDirPdfW, &cameraBsdfRevPdfW, 
        &aCameraPrd.launchIndex);
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  - cameraCosTheta  % 14f \n", cameraCosTheta);
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  - cameraBsdfFact  % 14f % 14f % 14f\n", 
        cameraBsdfFactor.x, cameraBsdfFactor.y, cameraBsdfFactor.z);

    if (isZero(cameraBsdfFactor))
    {
        OPTIX_PRINTFI(aCameraPrd.depth, "conn  - SKIP cam BSDF zero \n\n");
        return;
    }

    // Add camera continuation probability (for russian roulette)
    const float cameraCont = aCameraBsdf.continuationProb();
    cameraBsdfDirPdfW *= cameraCont;
    cameraBsdfRevPdfW *= cameraCont;

    // Evaluate BSDF at light vertex
    float lightCosTheta, lightBsdfDirPdfW, lightBsdfRevPdfW;
    const float3 lightBsdfFactor = alightVertex.bsdf.vcmF(-direction, lightCosTheta, &lightBsdfDirPdfW, &lightBsdfRevPdfW,
        &aCameraPrd.launchIndex);
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  - lightCosTheta   % 14f \n", lightCosTheta);
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  - lightBsdfFact   % 14f % 14f % 14f\n", lightBsdfFactor.x, lightBsdfFactor.y, lightBsdfFactor.z);

    // Geometry term
    const float geometryTerm = lightCosTheta * cameraCosTheta / dist2;
    const double dGeometryTerm = lightCosTheta * cameraCosTheta / double(dist2);
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  - gemoetryTerm    % 14e dGemometryTerm % 14e dist2         % 14f\n",
        geometryTerm, dGeometryTerm, dist2);

    if (geometryTerm < 0.f)
        return;

    // Convert solid angle pdfs to area pdfs
    const float cameraBsdfDirPdfA = PdfWtoA(cameraBsdfDirPdfW, distance, cameraCosTheta);
    const float lightBsdfDirPdfA = PdfWtoA(lightBsdfDirPdfW, distance, lightCosTheta);

    // Partial light sub-path MIS weight [tech. rep. (40)]
    const float wLight = vcmMis(cameraBsdfDirPdfA) * 
        ( misVmWeightFactor + alightVertex.dVCM + alightVertex.dVC * vcmMis(lightBsdfRevPdfW) );
    // lightBsdfRevPdfW is Reverse with respect to light path, e.g. in eye path progression 
    // dirrection (note same arrow dirs in formula)
    // note (40) and (41) uses light subpath Y and camera subpath z
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  - misVmFactor     % 14f\n", misVmWeightFactor);
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  - LightVertex dVC % 14e            dVM % 14e           dVCM % 14e\n",
        alightVertex.dVC, alightVertex.dVM, alightVertex.dVCM);
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  - camBsdfDirPdfA  % 14e camBsdfDirPdfW % 14f       distance % 14f cameraCosTheta % 14f\n",
        cameraBsdfDirPdfA, cameraBsdfDirPdfW, distance, cameraCosTheta);
    //OPTIX_PRINTFI(aCameraPrd.depth, "conn  -          wLight % 14f camBsdfDirPdfA % 14f lgtBsdfRevPdfW % 14f\n",
    //    wLight, cameraBsdfDirPdfA, lightBsdfRevPdfW);

    // Partial eye sub-path MIS weight [tech. rep. (41)]
    const float wCamera = vcmMis(lightBsdfDirPdfA) * 
        ( misVmWeightFactor + aCameraPrd.dVCM + aCameraPrd.dVC * vcmMis(cameraBsdfRevPdfW) );

    OPTIX_PRINTFI(aCameraPrd.depth, "conn  - Camera      dVC % 14e            dVM % 14e           dVCM % 14e\n",
        aCameraPrd.dVC, aCameraPrd.dVM, aCameraPrd.dVCM);
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  - lgtBsdfDirPdfA  % 14e lgtBsdfRevPdfW % 14f       distance % 14f  lightCosTheta % 14f\n",
        lightBsdfDirPdfA, lightBsdfRevPdfW, distance, lightCosTheta);
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  -          wLight % 14f camBsdfDirPdfA % 14f lgtBsdfRevPdfW % 14f\n",
        wLight, cameraBsdfDirPdfA, lightBsdfRevPdfW);

    // Full path MIS weight [tech. rep. (37)]
    const float misWeight = 1.f / (wLight + 1.f + wCamera);
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  -       misWeight % 14f         wLight % 14f        wCamera % 14f\n",
        wLight, wCamera, misWeight);
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  -  Cam througput  % 14f % 14f % 14f\n",
        aCameraPrd.throughput.x, aCameraPrd.throughput.z, aCameraPrd.throughput.y);
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  - Vert througput  % 14f % 14f % 14f\n",
        alightVertex.throughput.x, alightVertex.throughput.z, alightVertex.throughput.y);

    float3 contrib = (geometryTerm * cameraBsdfFactor * lightBsdfFactor) / alightVertexPickPdf; 
    contrib *= misWeight * aCameraPrd.throughput * alightVertex.throughput;

    OPTIX_PRINTFI(aCameraPrd.depth, "conn  - CONNECT contrib % 14e % 14e % 14e \n\n", contrib.x , contrib.y, contrib.z);
    if (isOccluded(aCameraHitpoint, direction, distance))
        return;
    
    aCameraPrd.color += contrib;
    //rtPrintf("%d %d - d %d - Contrib vetext at %f %f %f\n",
    //    aCameraPrd.launchIndex.x, aCameraPrd.launchIndex.y, aCameraPrd.depth, contrib.x, contrib.y, contrib.z);

    //Lambertian * lambertian = reinterpret_cast<Lambertian *>(aVertex.bsdf.bxdfAt(0));
    //float3 kd = lambertian->rho(1, NULL, NULL);
    //rtPrintf("%d %d Unoccluded vetext at %f %f %f dirFix %f %f %f Kd %f %f %f\n",
    //    aCameraPrd.launchIndex.x, aCameraPrd.launchIndex.y,
    //    aVertex.hitPoint.x, aVertex.hitPoint.y, aVertex.hitPoint.z,
    //    aVertex.bsdf.localDirFix().x, aVertex.bsdf.localDirFix().y, aVertex.bsdf.localDirFix().z,
    //    kd.x, kd.y, kd.z);
}




rtDeclareVariable(uint, vcmNumlightVertexConnections, , );

 // Camra subpath program
RT_PROGRAM void vcmClosestHitCamera()
{
    //OPTIX_PRINTFI(subpathPrd.depth, "CamHit\n");
    subpathPrd.depth++;	

    // vmarz TODO make sure shading normals used correctly
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;

    OPTIX_PRINTFI(subpathPrd.depth, "Hit C - incident dir W  % 14f % 14f % 14f\n", ray.direction.x, ray.direction.y, ray.direction.z);
    OPTIX_PRINTFI(subpathPrd.depth, "Hit C - hitPoint        % 14f % 14f % 14f\n", hitPoint.x, hitPoint.y, hitPoint.z);
    OPTIX_PRINTFI(subpathPrd.depth, "Hit C - normal W        % 14f % 14f % 14f\n", worldShadingNormal.x, worldShadingNormal.y, worldShadingNormal.z);

    // vmarz TODO infinite lights need attitional handling
    float cosThetaIn = dot(worldShadingNormal, -ray.direction);
    OPTIX_PRINTFI(subpathPrd.depth, "Hit C - cosThetaIn      % 14f \n", cosThetaIn);
    OPTIX_PRINTFI(subpathPrd.depth, "Hit C - incident dir W  % 14f % 14f % 14f\n", ray.direction.x, ray.direction.y, ray.direction.z);
    if (cosThetaIn < EPS_COSINE) // reject if cos too low
    {
        subpathPrd.done = 1;
        return;
    }   

    OPTIX_PRINTFI(subpathPrd.depth, "Hit C - cosThetaIn      % 14f         rayLen % 14f\n", cosThetaIn, tHit);
    OPTIX_PRINTFI(subpathPrd.depth, "Hit C - MIS preUpd  dVC % 14f            dVM % 14f           dVCM % 14f\n",
        subpathPrd.dVC, subpathPrd.dVM, subpathPrd.dVCM);
    updateMisTermsOnHit(subpathPrd, cosThetaIn, tHit);
    OPTIX_PRINTFI(subpathPrd.depth, "Hit C - MIS postUpd dVC % 14f            dVM % 14f           dVCM % 14f\n",
        subpathPrd.dVC, subpathPrd.dVM, subpathPrd.dVCM);

    VcmBSDF cameraBsdf;
    setVcmBSDF(cameraBsdf, worldShadingNormal, -ray.direction);
    // TODO connect to light source

    // Connect to ligth vertices // TODO move to func
    uint numLightVertices = lightVertexBufferIndexBuffer[0];
    float vertexPickPdf = float(vcmNumlightVertexConnections) / numLightVertices; // TODO scale by pick prob
    for (int i = 0; i < vcmNumlightVertexConnections; i++)
    {
        uint vertIdx = numLightVertices * getRandomUniformFloat(&subpathPrd.randomState);
        LightVertex lightVertex = lightVertexBuffer[vertIdx];
        connectVertices(lightVertex, vertexPickPdf, cameraBsdf, subpathPrd, hitPoint);
    }
    
    // vmarz TODO check max path length
    // Russian Roulette
    float contProb =  cameraBsdf.continuationProb();// luminanceCIE(Kd); // vmarz TODO precompute
    float rrSample = getRandomUniformFloat(&subpathPrd.randomState);    
    OPTIX_PRINTFI(subpathPrd.depth, "Hit C - continue sample % 14f             RR % 14f \n", contProb, rrSample);
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
    //OPTIX_PRINTFI(subpathPrd.depth, "Hit - new dir %f %f %f\n", subpathPrd.direction.x, subpathPrd.direction.y, subpathPrd.direction.z);

    float bsdfRevPdfW = cosThetaIn * M_1_PIf;
    bsdfDirPdfW *= contProb;
    bsdfRevPdfW *= contProb;
    updateMisTermsOnScatter(subpathPrd, cosThetaOut, bsdfDirPdfW, bsdfRevPdfW, misVcWeightFactor, misVmWeightFactor);

    // f * cosTheta / f_pdf
    subpathPrd.throughput *= bsdfFactor * (cosThetaOut / bsdfDirPdfW);
    subpathPrd.origin = hitPoint;
    OPTIX_PRINTFI(subpathPrd.depth, "Hit C - new origing     % 14f % 14f % 14f\n", 
      subpathPrd.origin.x, subpathPrd.origin.y, subpathPrd.origin.z);
}