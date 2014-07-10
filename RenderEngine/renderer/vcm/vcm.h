#pragma once

//#define OPTIX_PRINTF_DEF
//#define OPTIX_PRINTFI_DEF
//#define OPTIX_PRINTFID_DEF

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
#include "material/BSDF.h"
#include "renderer/vcm/LightVertex.h"
#include "renderer/vcm/SubpathPRD.h"
#include "renderer/vcm/config_vcm.h"
#include "renderer/vcm/mis.h"

#define OPTIX_PRINTF_ENABLED 1
#define OPTIX_PRINTFI_ENABLED 1
#define OPTIX_PRINTFID_ENABLED 1

RT_FUNCTION int isOccluded(const rtObject      & aSceneRootObject, 
                           const optix::float3 & aPoint, 
                           const optix::float3 & aDirection, 
                           const float           aTMax)
{
    using namespace optix;
    if (aTMax < 3.f*EPS_RAY)
        return false;

    ShadowPRD shadowPrd;
    shadowPrd.attenuation = 1.0f;
    Ray occlusionRay(aPoint, aDirection, RayType::SHADOW, EPS_RAY, aTMax - 2.f*EPS_RAY);

    //OPTIX_PRINTF("isOccluded aPoint %f %f %f direction %f %f %f dist %f \n", 
    //    aPoint.x, aPoint.y, aPoint.z, aDirection.x, aDirection.y, aDirection.z, aTMax);
    rtTrace(aSceneRootObject, occlusionRay, shadowPrd);
    return shadowPrd.attenuation == 0.f;
}



RT_FUNCTION void lightHit( SubpathPRD                  & aLightPrd,
                          const optix::float3          & aHitPoint, 
                          const optix::float3          & aWorldNormal,
                          const optix::float3            aRayWorldDir,  // not passing ray dir by reference sine it's OptiX semantic type
                          const float                    aRayTHit,
                          const optix::uint              aLightVertexCountEstimatePass,
                          const float                    aMisVcWeightFactor,
                          const float                    aMisVmWeightFactor,
                          rtBufferId<LightVertex>        aLightVertexBuffer,
                          rtBufferId<optix::uint>        aLightVertexBufferIndexBuffer,
#if !VCM_UNIFORM_VERTEX_SAMPLING                         // for 1 to 1 camera - light path connections
                          rtBufferId<optix::uint, 3>     aLightSubpathVertexIndexBuffer,
                          const optix::uint              aLightSubpathMaxLen,
#else                                                    // uniform vertex sampling
                          const float                  * aVertexPickPdf,
#endif
                          const BxDF                   * bxdf1,
                          const BxDF                   * bxdf2 = NULL )
{
    using namespace optix;

#if VCM_UNIFORM_VERTEX_SAMPLING
    const float *pVertexPickPdf = aVertexPickPdf;
#else
    const float *pVertexPickPdf = NULL;
#endif

    aLightPrd.depth++;
    OPTIX_PRINTFID(aLightPrd.launchIndex, aLightPrd.depth, "Hit L - incident dir W  % 14f % 14f % 14f\n", aRayWorldDir.x, aRayWorldDir.y, aRayWorldDir.z);
    OPTIX_PRINTFID(aLightPrd.launchIndex, aLightPrd.depth, "Hit L - hitPoint        % 14f % 14f % 14f\n", aHitPoint.x, aHitPoint.y, aHitPoint.z);
    OPTIX_PRINTFID(aLightPrd.launchIndex, aLightPrd.depth, "Hit L - normal W        % 14f % 14f % 14f\n", aWorldNormal.x, aWorldNormal.y, aWorldNormal.z);

    // vmarz TODO infinite lights need additional handling
    float cosThetaIn = dot(aWorldNormal, -aRayWorldDir);
    OPTIX_PRINTFID(aLightPrd.launchIndex, aLightPrd.depth, "Hit L - cos theta       % 14f \n", cosThetaIn);
    if (cosThetaIn < EPS_COSINE) // reject if cos too low
    {
        aLightPrd.done = 1;
        return;
    }   

    updateMisTermsOnHit(aLightPrd, cosThetaIn, aRayTHit);

    LightVertex lightVertex;
    lightVertex.launchIndex = aLightPrd.launchIndex;
    lightVertex.hitPoint = aHitPoint;
    lightVertex.throughput = aLightPrd.throughput;
    lightVertex.pathLen = aLightPrd.depth;
    lightVertex.dVCM = aLightPrd.dVCM;
    lightVertex.dVC = aLightPrd.dVC;
    lightVertex.dVM = aLightPrd.dVM;
#if VCM_UNIFORM_VERTEX_SAMPLING
    lightVertex.dVC = aLightPrd.dVC_unif_vert;
    // There is no dVC_unif_vert in LightVertex since vertices are used only for connection between each other,
    // and do not affect connection to camera/light source and dVC is not present in weight equation for VM.
    // equations in [tech. rep. (38-47)]
#endif
    lightVertex.bsdf = VcmBSDF(aWorldNormal, -aRayWorldDir);
    lightVertex.bsdf.AddBxDF(bxdf1);
    if (bxdf2) lightVertex.bsdf.AddBxDF(bxdf2);

    DifferentialGeometry dg = lightVertex.bsdf.differentialGeometry();
    //OPTIX_PRINTFID(aLightPrd.launchIndex, aLightPrd.depth, "Hit L - frame vectors b % 14f % 14f % 14f\n", dg.bitangent.x, dg.bitangent.y, dg.bitangent.z);
    //OPTIX_PRINTFID(aLightPrd.launchIndex, aLightPrd.depth, "Hit L -               t % 14f % 14f % 14f\n", dg.tangent.x, dg.tangent.y, dg.tangent.z);
    //OPTIX_PRINTFID(aLightPrd.launchIndex, aLightPrd.depth, "Hit L -               n % 14f % 14f % 14f\n", dg.normal.x, dg.normal.y, dg.normal.z);
    float3 dirFix = lightVertex.bsdf.localDirFix();
    OPTIX_PRINTFID(aLightPrd.launchIndex, aLightPrd.depth, "Hit L - dir fix local   % 14f % 14f % 14f\n", dirFix.x, dirFix.y, dirFix.z);

    // store path vertex
    if (!aLightVertexCountEstimatePass)
    {
#if !VCM_UNIFORM_VERTEX_SAMPLING
        if (aLightPrd.depth == aLightSubpathMaxLen)
        {
            OPTIX_PRINTF("%u %u - d %u - Hit L - Light path reached MAX LENGTH \n", aLightPrd.launchIndex.x, aLightPrd.launchIndex.y, aLightPrd.depth);
            aLightPrd.done = 1;
            return;
        }
#endif
        //uint vertIdx = atomicAdd(&aLightVertexBufferIndexBuffer[0], 1u);
        uint vertIdx = atomicAdd(&aLightVertexBufferIndexBuffer[0], 1u);
        OPTIX_PRINTFID(aLightPrd.launchIndex, aLightPrd.depth, "Hit L - Vert.throuhput  % 14f % 14f % 14f\n", 
            lightVertex.throughput.x, lightVertex.throughput.y, lightVertex.throughput.z);
        aLightVertexBuffer[vertIdx] = lightVertex;

#if !VCM_UNIFORM_VERTEX_SAMPLING
        //uint3 pathVertIdx = make_uint3(launchIndex, aLightPrd.depth-1); // getting this ?? 1072693248 0 0 or 1072693248 1 0
        uint3 pathVertIdx = make_uint3(aLightPrd.launchIndex.x, aLightPrd.launchIndex.y, aLightPrd.depth-1);
        //OPTIX_PRINTF("%d %d - d %d - Hit L- Store VertIdx %u v.pathLen %d pathVertIdx %u %u %u\n",
        //    launchIndex.x, launchIndex.y, aLightPrd.depth, vertIdx, lightVertex.pathLen, 
        //    pathVertIdx.x, pathVertIdx.y, pathVertIdx.z);
        aLightSubpathVertexIndexBuffer[pathVertIdx] = vertIdx;
#endif
    }
    
    // vmarz TODO connect to camera

    // Russian Roulette
    float contProb =  lightVertex.bsdf.continuationProb();
    float rrSample = getRandomUniformFloat(&aLightPrd.randomState);    
    OPTIX_PRINTFID(aLightPrd.launchIndex, aLightPrd.depth, "Hit L - continue sample % 14f             RR % 14f \n", contProb, rrSample);
    if (contProb < rrSample)
    {
        aLightPrd.done = 1;
        return;
    }

    //next event
    float bsdfDirPdfW;
    float cosThetaOut;
    float3 bsdfSample = getRandomUniformFloat3(&aLightPrd.randomState);
    float3 bsdfFactor = lightVertex.bsdf.vcmSampleF(&aLightPrd.direction, bsdfSample, &bsdfDirPdfW, &cosThetaOut); // CUDA 6 fails here
    OPTIX_PRINTFID(aLightPrd.launchIndex, aLightPrd.depth, "Hit L - new dir World   % 14f % 14f % 14f\n",
        aLightPrd.direction.x, aLightPrd.direction.y, aLightPrd.direction.z);
    OPTIX_PRINTFID(aLightPrd.launchIndex, aLightPrd.depth, "Hit L -      bsdfFactor % 14f % 14f % 14f \n", bsdfFactor.x, bsdfFactor.y, bsdfFactor.z);

    if (isZero(bsdfFactor))
        return;

    float bsdfRevPdfW = cosThetaIn * M_1_PIf;
    bsdfDirPdfW *= contProb;
    bsdfRevPdfW *= contProb;
    //if (isNAN(dVC))
    //{
    //    OPTIX_PRINTF("%d %d - d %d - MIS - DQNAN aPathPrd.dVC - aCosThetaOut %f  cosDivBsfPdf %f cosDivBsfPdfD %f \n", 
    //        aPathPrd.launchIndex.x, aPathPrd.launchIndex.y, aPathPrd.depth, aCosThetaOut, cosDivBsfPdf, cosDivBsfPdfD);
    //    OPTIX_PRINTF("%d %d - d %d - MIS- DQNAN aPathPrd.dVC - aBsdfDirPdfW %f  cosDivBsfPdf %f cosDivBsfPdfD %f \n", 
    //        aPathPrd.launchIndex.x, aPathPrd.launchIndex.y, aPathPrd.depth, aBsdfDirPdfW, cosDivBsfPdf, cosDivBsfPdfD);
    //}
    updateMisTermsOnScatter(aLightPrd, cosThetaOut, bsdfDirPdfW, bsdfRevPdfW, aMisVcWeightFactor, aMisVmWeightFactor, pVertexPickPdf);

    OPTIX_PRINTFID(aLightPrd.launchIndex, aLightPrd.depth, "Hit L - prd.througput1  % 14f % 14f % 14f \n", 
        aLightPrd.throughput.x, aLightPrd.throughput.y, aLightPrd.throughput.z);
    OPTIX_PRINTFID(aLightPrd.launchIndex, aLightPrd.depth, "Hit L - th=(cosThetaOut % 14f /  bsdfDirPdfW % 14f ) * througput * bsdfactor \n",
        cosThetaOut, bsdfDirPdfW);

    // f * cosTheta / f_pdf
    aLightPrd.throughput *= bsdfFactor * (cosThetaOut / bsdfDirPdfW); 
    OPTIX_PRINTFID(aLightPrd.launchIndex, aLightPrd.depth, "Hit L - prd.througput2  % 14f % 14f % 14f \n", 
        aLightPrd.throughput.x, aLightPrd.throughput.y, aLightPrd.throughput.z);

    aLightPrd.origin = aHitPoint;
    OPTIX_PRINTFID(aLightPrd.launchIndex, aLightPrd.depth, "Hit L - new origin      % 14f % 14f % 14f\n\n", 
        aLightPrd.origin.x, aLightPrd.origin.y, aLightPrd.origin.z);
}




// Connects vertices and return contribution
RT_FUNCTION void connectVertices( const rtObject        & aSceneRootObject,
                                  const LightVertex     & aLightVertex,
                                  SubpathPRD            & aCameraPrd,
                                  const VcmBSDF         & aCameraBsdf,
                                  const optix::float3   & aCameraHitpoint,
                                  const float             aMisVmWeightFactor,
                                  const float const     * aVertexPickPdf = NULL )
{
    using namespace optix;

    uint2 launchIndex = aCameraPrd.launchIndex;
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "connectVertices(): \n");
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  -  cameraHitPoint % 14f % 14f % 14f \n",
        aCameraHitpoint.x, aCameraHitpoint.y, aCameraHitpoint.z);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  - --> vertex      % 14f % 14f % 14f \n",
        aLightVertex.hitPoint.x, aLightVertex.hitPoint.y, aLightVertex.hitPoint.z);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  -         pathLen % 14u            id %5u %5u \n",
        aLightVertex.pathLen, aLightVertex.launchIndex.x, aLightVertex.launchIndex.y);

    // Get connection
    float3 direction = aLightVertex.hitPoint - aCameraHitpoint;
    float dist2      = dot(direction, direction);
    float distance   = sqrt(dist2);
    direction       /= distance;
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  -             dir % 14f % 14f % 14f           dist % 14f\n",
        direction.x, direction.y, direction.z, distance);

    // Evaluate BSDF at camera vertex
    float cameraCosTheta, cameraBsdfDirPdfW, cameraBsdfRevPdfW;
    const float3 cameraBsdfFactor = aCameraBsdf.vcmF(direction, cameraCosTheta, &cameraBsdfDirPdfW, &cameraBsdfRevPdfW, 
        &aCameraPrd.launchIndex);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  -  cameraCosTheta % 14f \n", cameraCosTheta);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  -  cameraBsdfFact % 14f % 14f % 14f\n", 
        cameraBsdfFactor.x, cameraBsdfFactor.y, cameraBsdfFactor.z);

    if (isZero(cameraBsdfFactor))
    {
        OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  - SKIP connect Camera BSDF zero \n");
        return;
    }

    // Add camera continuation probability (for russian roulette)
    const float cameraCont = aCameraBsdf.continuationProb();
    cameraBsdfDirPdfW *= cameraCont;
    cameraBsdfRevPdfW *= cameraCont;

    // Evaluate BSDF at light vertex
    float lightCosTheta, lightBsdfDirPdfW, lightBsdfRevPdfW;
    const float3 lightBsdfFactor = aLightVertex.bsdf.vcmF(-direction, lightCosTheta, &lightBsdfDirPdfW, &lightBsdfRevPdfW,
        &aCameraPrd.launchIndex);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  -   lightCosTheta % 14f \n", lightCosTheta);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  -   lightBsdfFact % 14f % 14f % 14f\n", lightBsdfFactor.x, lightBsdfFactor.y, lightBsdfFactor.z);
    
    if (isZero(lightBsdfFactor))
    {
        OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  - SKIP connect Light BSDF zero \n");
        return;
    }

    // Geometry term
    const float geometryTerm = lightCosTheta * cameraCosTheta / dist2;
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  -    geometryTerm % 14f         dist2 % 14f\n", geometryTerm, dist2);

    if (geometryTerm < 0.f)
        return;

    // Convert solid angle pdfs to area pdfs
    const float cameraBsdfDirPdfA = PdfWtoA(cameraBsdfDirPdfW, distance, cameraCosTheta);
    const float lightBsdfDirPdfA = PdfWtoA(lightBsdfDirPdfW, distance, lightCosTheta);

    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  - camBsdfDirPdfA = (camBsdfDirPdfW *       cosLight) / sqr (      distance) \n");
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  - % 14f = (% 14f * % 14f) / sqr (% 14f) \n",
        cameraBsdfDirPdfA, cameraBsdfDirPdfW, cameraCosTheta, distance);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  - lgtBsdfDirPdfA = (lgtBsdfDirPdfW *      cosCamera) / sqr (      distance) \n");
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  - % 14f = (% 14f * % 14f) / sqr (% 14f) \n",
        lightBsdfDirPdfA, lightBsdfDirPdfW, lightCosTheta, distance);

    // aVertPickPdf is set only when unform vertex sampling used (connecting to all paths)
    float invVertPickPdf = aVertexPickPdf ? (1.f / *aVertexPickPdf) : 1.f;
    float aCameraPrd_dVC = aCameraPrd.dVC;
#if VCM_UNIFORM_VERTEX_SAMPLING
    aCameraPrd_dVC = aCameraPrd.dVC_unif_vert;
    // There is no dVC_unif_vert in LightVertex since vertices are used only for connection between each other,
    // and do not affect connection to camera/light source and dVC is not present in weight equation for VM.
    // equations in [tech. rep. (38-47)]
    OPTIX_PRINTFI(aCameraPrd.depth, "conn  -  invVertPickPdf % 14f \n", invVertPickPdf);
#endif

    // Partial light sub-path MIS weight [tech. rep. (40)]
    const float wLight = vcmMis(cameraBsdfDirPdfA) * 
        ( aMisVmWeightFactor * invVertPickPdf + aLightVertex.dVCM + aLightVertex.dVC * vcmMis(lightBsdfRevPdfW) );
    // lightBsdfRevPdfW is Reverse with respect to light path, e.g. in eye path progression 
    // dirrection (note same arrow dirs in formula)
    // note (40) and (41) uses light subpath Y and camera subpath z;
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  - LightVertex dVC % 14e            dVM % 14e           dVCM % 14e\n",
        aLightVertex.dVC, aLightVertex.dVM, aLightVertex.dVCM);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  -         wLight = camBsdfDirPdfA * (VmWeightFactor +     light.dVCM +      light.dVC * lgtBsdfRevPdfW) \n");
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  - % 14f = % 14f * (% 14f + % 14e + % 14f * % 14f) \n", 
        wLight, cameraBsdfDirPdfA, aMisVmWeightFactor, aLightVertex.dVCM, aLightVertex.dVC, lightBsdfRevPdfW);

    // Partial eye sub-path MIS weight [tech. rep. (41)]
    const float wCamera = vcmMis(lightBsdfDirPdfA) * 
        ( aMisVmWeightFactor * invVertPickPdf + aCameraPrd.dVCM + aCameraPrd_dVC * vcmMis(cameraBsdfRevPdfW) );
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  - Camera      dVC % 14e            dVM % 14e           dVCM % 14e\n",
        aCameraPrd_dVC, aCameraPrd.dVM, aCameraPrd.dVCM);    
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  -        wCamera = lgtBsdfDirPdfA * (VmWeightFactor +    camera.dVCM +     camera.dVC * camBsdfRevPdfW) \n");
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  - % 14f = % 14f * (% 14f + % 14e + % 14f * % 14f) \n", 
        wLight, lightBsdfDirPdfA, aMisVmWeightFactor, aCameraPrd.dVCM, aCameraPrd.dVC, cameraBsdfRevPdfW);

    // Full path MIS weight [tech. rep. (37)]
    const float misWeight = 1.f / (wLight + 1.f + wCamera);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  -       misWeight % 14f         wLight % 14f        wCamera % 14f\n",
        misWeight, wLight, wCamera);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  -  Cam througput  % 14f % 14f % 14f\n",
        aCameraPrd.throughput.x, aCameraPrd.throughput.z, aCameraPrd.throughput.y);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  - Vert througput  % 14f % 14f % 14f\n",
        aLightVertex.throughput.x, aLightVertex.throughput.z, aLightVertex.throughput.y);

    float3 contrib = geometryTerm * cameraBsdfFactor * lightBsdfFactor * invVertPickPdf;
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  - noThp unw cntrb % 14f % 14f % 14f \n", contrib.x, contrib.y, contrib.z);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  - noThp wei cntrb = geometryTerm * cameraBsdfFactor * lightBsdfFactor * invVertPickPdf \n");
    contrib *= misWeight;
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  - noThp wei cntrb % 14f % 14f % 14f \n", contrib.x, contrib.y, contrib.z);
    contrib *= aCameraPrd.throughput * aLightVertex.throughput;
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  -   Thp wei cntrb % 14f % 14f % 14f \n", contrib.x, contrib.y, contrib.z);

    //if (isNAN(aCameraPrd.color.x))
    //    OPTIX_PRINTF("%d %d - d %d - DQNAN prd.color  \n", launchIndex.x, launchIndex.y, aCameraPrd.depth);
    //if (isNAN(invVertPickPdf))
    //    OPTIX_PRINTF("%d %d - d %d - DQNAN invVertPickPdf  \n", launchIndex.x, launchIndex.y, aCameraPrd.depth);
    //if (isNAN(misWeight))
    //    OPTIX_PRINTF("%d %d - d %d - DQNAN misWeight  \n", launchIndex.x, launchIndex.y, aCameraPrd.depth);
    //if (isNAN(wLight))
    //    OPTIX_PRINTF("%d %d - d %d - DQNAN wLight  \n", launchIndex.x, launchIndex.y, aCameraPrd.depth);
    //if (isNAN(wCamera))
    //    OPTIX_PRINTF("%d %d - d %d - DQNAN wCamera  \n", launchIndex.x, launchIndex.y, aCameraPrd.depth);
    //if (isNAN(lightBsdfDirPdfA))
    //    OPTIX_PRINTF("%d %d - d %d - DQNAN lightBsdfDirPdfA \n", launchIndex.x, launchIndex.y, aCameraPrd.depth);
    //if (isNAN(aCameraPrd_dVC))
    //    OPTIX_PRINTF("%d %d - d %d - DQNAN aCameraPrd_dVC  \n", launchIndex.x, launchIndex.y, aCameraPrd.depth);
    //if (isNAN(cameraBsdfRevPdfW))
    //    OPTIX_PRINTF("%d %d - d %d - DQNAN cameraBsdfRevPdfW  \n", launchIndex.x, launchIndex.y, aCameraPrd.depth);
    //if (isNAN(lightBsdfDirPdfW))
    //    OPTIX_PRINTF("%d %d - d %d - DQNAN lightBsdfDirPdfW  \n", launchIndex.x, launchIndex.y, aCameraPrd.depth);
    //if (isNAN(lightCosTheta))
    //    OPTIX_PRINTF("%d %d - d %d - DQNAN lightCosTheta  \n", launchIndex.x, launchIndex.y, aCameraPrd.depth);

    //OPTIX_PRINTF("%d %d - d %d - Occlude check aCameraHitpoint %f %f %f \n", 
    //    launchIndex.x, launchIndex.y, aCameraPrd.depth, aCameraHitpoint.x, aCameraHitpoint.y, aCameraHitpoint.z);
    //OPTIX_PRINTF("%d %d - d %d - Occlude check direction %f %f %f dist %f\n", 
    //    launchIndex.x, launchIndex.y, aCameraPrd.depth, direction.x, direction.y, direction.z, distance);
    //OPTIX_PRINTF("%d %d - d %d - Occlude check direction %f %f %f \n", 
    //    launchIndex.x, launchIndex.y, aCameraPrd.depth, direction.x, direction.y, direction.z);

    // TODO try early occlusion check
    if (isOccluded(aSceneRootObject, aCameraHitpoint, direction, distance)) // CRASH HERE
    {
        OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  - OCCLUDED\n");
        return;
    }
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "\n");

    aCameraPrd.color += contrib;
}



// Connects camera subpath vertex to light source, e.g. direct illumination, next event estimation.
// Light subpath length=1 [tech. rep 44-45]
RT_FUNCTION void connectLightSourceS1( const rtObject             & aSceneRootObject,
                                       const rtBufferId<Light, 1>   alightsBuffer,
                                       SubpathPRD                 & aCameraPrd,
                                       const VcmBSDF              & aCameraBsdf,
                                       const optix::float3        & aCameraHitpoint,
                                       const float                  aMisVmWeightFactor,
                                       const float const          * aVertexPickPdf = NULL)
{
    using namespace optix;
    uint2 launchIndex = aCameraPrd.launchIndex;

    int lightIndex = 0;
    if (1 < alightsBuffer.size())
    {
        float sample = getRandomUniformFloat(&aCameraPrd.randomState);
        lightIndex = intmin((int)(sample*alightsBuffer.size()), int(alightsBuffer.size()-1));
    }

    const Light light               = alightsBuffer[lightIndex];
    const float inverseLightPickPdf = alightsBuffer.size();
    const float lightPickProb        = 1.f / alightsBuffer.size();

    float emissionPdfW;
    float directPdfW;
    float cosAtLight;
    float distance;
    float3 dirToLight;
    float3 radiance = lightIlluminate(light, aCameraPrd.randomState, aCameraHitpoint, dirToLight,
        distance, emissionPdfW, &directPdfW, &cosAtLight, &aCameraPrd.launchIndex);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "connL-  light radiance % 14f % 14f % 14f \n", radiance.x, radiance.y, radiance.z);
    
    if (isZero(radiance))
        return;

    float bsdfDirPdfW, bsdfRevPdfW, cosToLight;
    float3 bsdfFactor = aCameraBsdf.vcmF(dirToLight, cosToLight, &bsdfDirPdfW, &bsdfRevPdfW);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "connL-      bsdfFactor % 14f % 14f % 14f \n", bsdfFactor.x, bsdfFactor.y, bsdfFactor.z);

    if (isZero(bsdfFactor))
    {
        OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "connL- ZERO bsdfFactor\n");
        return;
    }

    const float contiueProb = aCameraBsdf.continuationProb();

    // If the light is delta light, we can never hit it by BSDF sampling, so the probability of this path is 0
    bsdfDirPdfW *= light.isDelta ? 0.f : contiueProb;
    bsdfRevPdfW *= contiueProb;

    // Partial light sub-path MIS weight [tech. rep. (44)].
    // Note that wLight is a ratio of area pdfs. But since both are on the
    // light source, their distance^2 and cosine terms cancel out.
    // Therefore we can write wLight as a ratio of solid angle pdfs,
    // both expressed w.r.t. the same shading point.
    const float wLight = vcmMis(bsdfDirPdfW / (lightPickProb * directPdfW));

    // Partial eye sub-path MIS weight [tech. rep. (45)].
    //
    // In front of the sum in the parenthesis we have Mis(ratio), where
    //    ratio = emissionPdfA / directPdfA,
    // with emissionPdfA being the product of the pdfs for choosing the
    // point on the light source and sampling the outgoing direction.
    // What we are given by the light source instead are emissionPdfW
    // and directPdfW. Converting to area pdfs and plugging into ratio:
    //    emissionPdfA = emissionPdfW * cosToLight / dist^2
    //    directPdfA   = directPdfW * cosAtLight / dist^2
    //    ratio = (emissionPdfW * cosToLight / dist^2) / (directPdfW * cosAtLight / dist^2)
    //    ratio = (emissionPdfW * cosToLight) / (directPdfW * cosAtLight)
    //
    // Also note that both emissionPdfW and directPdfW should be
    // multiplied by lightPickProb, so it cancels out.
    const float wCamera = vcmMis(emissionPdfW * cosToLight / (directPdfW * cosAtLight)) *
        (aMisVmWeightFactor + aCameraPrd.dVCM + aCameraPrd.dVC * vcmMis(bsdfRevPdfW));

    // Full path MIS weight [tech. rep. (37)]
    const float misWeight = 1.f / (wLight + 1.f + wCamera);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "connL-       misWeight % 14f         wLight % 14f        wCamera % 14f\n",
        misWeight, wLight, wCamera);

    float3 contrib = (misWeight * cosToLight / (lightPickProb * directPdfW)) * (radiance * bsdfFactor);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "connL- noThp wei cntrb % 14f % 14f % 14f \n", contrib.x, contrib.y, contrib.z);

    if (isZero(contrib) || isOccluded(aSceneRootObject, aCameraHitpoint, dirToLight, distance))
        return;

    contrib *= aCameraPrd.throughput;
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "connL-   Thp wei cntrb % 14f % 14f % 14f \n", contrib.x, contrib.y, contrib.z);

    aCameraPrd.color += contrib;
}


// Computes vertex connection contribution for camera path vertex that has been sampled on 
// light source (traced and randomly hit the light). Light subpath length = 0 [tech. rep 42-43]
RT_FUNCTION void connectLightSourceS0(SubpathPRD aCameraPrd, const optix::float3 &aRadiance, float & aDirectPdfA, 
                                     float & aEmissionPdfW, float aLightPickProb, int aUseVC, int aUseVM)
{
    if (aCameraPrd.depth == 1) // first hit, see directly from camera, no weighting needed
    {
        aCameraPrd.color += aCameraPrd.throughput * aRadiance;
        return;
    }

    // TODO specular path
    //if (!vcmUseVC && aUseVM)
    //{
    //    if (aCameraPrd.specularPath)
    //        aCameraPrd.color += aRadiance;
    //    return;
    //}

    aDirectPdfA *= aLightPickProb;
    aEmissionPdfW *= aLightPickProb;

    // Partial eye sub-path MIS weight [tech. rep. (43)].
    // If the last hit was specular, then dVCM == 0.
    const float wCamera = vcmMis(aDirectPdfA) * aCameraPrd.dVCM + vcmMis(aEmissionPdfW * aCameraPrd.dVC);
    // Partial light sub-path weight is 0 [tech. rep. (42)].

    // Full path MIS weight [tech. rep. (37)].
    const float misWeight = 1.f / (1.f + wCamera);

    aCameraPrd.color += aCameraPrd.throughput * misWeight * aRadiance;
}



RT_FUNCTION void cameraHit( const rtObject                     & aSceneRootObject,
                            SubpathPRD                         & aCameraPrd,
                            const optix::float3                & aHitPoint,
                            const optix::float3                & aWorldNormal,
                            const optix::float3                  aRayWorldDir,  // not passing ray dir by reference sine it's OptiX semantic type
                            const float                          aRayTHit,
                            const float                          aMisVcWeightFactor,
                            const float                          aMisVmWeightFactor,
                            const rtBufferId<Light, 1>           alightsBuffer,
                            const rtBufferId<optix::uint, 2>     aLightSubpathLengthBuffer,
                            const rtBufferId<LightVertex>        aLightVertexBuffer,
                            const rtBufferId<optix::uint>        aLightVertexBufferIndexBuffer,
#if !VCM_UNIFORM_VERTEX_SAMPLING                                // for 1 to 1 camera - light path connections
                            const rtBufferId<optix::uint, 3>     aLightSubpathVertexIndexBuffer,
#else                                                           // uniform vertex sampling
                            const float                          averageLightSubpathLength,
                            const float                        * aVertexPickPdf,
#endif
                            const BxDF                         * bxdf1,
                            const BxDF                         * bxdf2 = NULL )
{
    using namespace optix;
    uint2 launchIndex = aCameraPrd.launchIndex;

#if VCM_UNIFORM_VERTEX_SAMPLING
    const float *pVertexPickPdf = aVertexPickPdf;
#else
    const float *pVertexPickPdf = NULL;
#endif

    //OPTIX_PRINTFI(aCameraPrd.depth, "CamHit\n");
    aCameraPrd.depth++;	
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "Hit C - hitPoint        % 14f % 14f % 14f\n", aHitPoint.x, aHitPoint.y, aHitPoint.z);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "Hit C - normal W        % 14f % 14f % 14f\n", aWorldNormal.x, aWorldNormal.y, aWorldNormal.z);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "Hit C - incident dir W  % 14f % 14f % 14f\n", aRayWorldDir.x, aRayWorldDir.y, aRayWorldDir.z);

    // vmarz TODO infinite lights need additional handling
    float cosThetaIn = dot(aWorldNormal, -aRayWorldDir);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "Hit C - cosThetaIn      % 14f \n", cosThetaIn);
    if (cosThetaIn < EPS_COSINE) // reject if cos too low
    {
        aCameraPrd.done = 1;
        return;
    }   

    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "Hit C - cosThetaIn      % 14f         rayLen % 14f\n", cosThetaIn, aRayTHit);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "Hit C - MIS preUpd  dVC % 14e            dVM % 14e           dVCM % 14e\n",
        aCameraPrd.dVC, aCameraPrd.dVM, aCameraPrd.dVCM);
    updateMisTermsOnHit(aCameraPrd, cosThetaIn, aRayTHit);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "Hit C - MIS postUpd dVC % 14e            dVM % 14e           dVCM % 14e\n",
        aCameraPrd.dVC, aCameraPrd.dVM, aCameraPrd.dVCM);

    VcmBSDF cameraBsdf = VcmBSDF(aWorldNormal, -aRayWorldDir);
    cameraBsdf.AddBxDF(bxdf1);
    if (bxdf2) cameraBsdf.AddBxDF(bxdf2);

    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "Hit C-  conn Lgt1 color % 14f % 14f % 14f \n",
        aCameraPrd.color.x, aCameraPrd.color.y, aCameraPrd.color.z);
    //connectLightSource(aSceneRootObject, alightsBuffer, aCameraPrd, cameraBsdf, aHitPoint, aMisVmWeightFactor);
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "Hit C-  conn Lgt2 color % 14f % 14f % 14f \n",
        aCameraPrd.color.x, aCameraPrd.color.y, aCameraPrd.color.z);

    // Connect to light vertices // TODO move to func
#if VCM_UNIFORM_VERTEX_SAMPLING
    uint numLightVertices = aLightVertexBufferIndexBuffer[0];
    //float vertexPickPdf = float(vcmNumlightVertexConnections) / numLightVertices; // TODO scale by pick prob
    uint numlightVertexConnections = ceilf(averageLightSubpathLength);
    float lastVertConnectProb = averageLightSubpathLength - (uint)averageLightSubpathLength;
    OPTIX_PRINTFI(aCameraPrd.depth, "Hit C - CONNECT     num % 14u   lastVertProb % 14f \n", 
        numlightVertexConnections, lastVertConnectProb);
    //for (int i = 0; i < vcmNumlightVertexConnections; i++)
    for (int i = 0; i < numlightVertexConnections; i++)
    {
        // For last vertex do russian roulette
        if (i == (numlightVertexConnections - 1))
        {
            float sampleConnect = getRandomUniformFloat(&aCameraPrd.randomState);
            if (lastVertConnectProb < sampleConnect)
                break;
        }

        uint vertIdx = numLightVertices * getRandomUniformFloat(&aCameraPrd.randomState);
        LightVertex lightVertex = aLightVertexBuffer[vertIdx];
        connectVertices(aSceneRootObject, lightVertex, aCameraPrd, cameraBsdf, aHitPoint, aMisVmWeightFactor, pVertexPickPdf);
    }
#else
    // CAUTION: this loop can cause weird issues, out of bound access with crazy indices, though they are based 
    // failing on launch index and loop variable, rtTrace crashing within the loop etc.
    // update: It seems it was caused multiple uses of std::printf
    uint lightSubpathLen = aLightSubpathLengthBuffer[launchIndex];
    //OPTIX_PRINTF("%d %d - d %d - Hit C-  conn lightSubpathLen %u \n", 
    //    launchIndex.x, launchIndex.y, aCameraPrd.depth, lightSubpathLen);
    for (uint i = 0; i < lightSubpathLen; ++i)
    {
        uint3 pathVertIdx = make_uint3(launchIndex.x, launchIndex.y, i);
        OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "%u %u - d %u - Hit C- %u conn pathVertIdx %u %u %u \n",
            launchIndex.x, launchIndex.y, aCameraPrd.depth, i, pathVertIdx.x, pathVertIdx.y, pathVertIdx.z);
        uint vertIdx = aLightSubpathVertexIndexBuffer[pathVertIdx];
        OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "%u %u - d %u - Hit C- %u conn vertIdx %u \n",
            launchIndex.x, launchIndex.y, aCameraPrd.depth, i, vertIdx);
        LightVertex lightVertex = aLightVertexBuffer[vertIdx];
        connectVertices(aSceneRootObject, lightVertex, aCameraPrd, cameraBsdf, aHitPoint, aMisVmWeightFactor);
        OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "Hit C-  conn Ver%u color % 14f % 14f % 14f \n",
            i, aCameraPrd.color.x, aCameraPrd.color.y, aCameraPrd.color.z);
    }    
#endif

    // vmarz TODO check max path length
    // Russian Roulette
    float contProb =  cameraBsdf.continuationProb();
    float rrSample = getRandomUniformFloat(&aCameraPrd.randomState);    
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "Hit C - continue sample % 14f             RR % 14f \n", contProb, rrSample);
    if (contProb < rrSample)
    {
        aCameraPrd.done = 1;
        return;
    }

    // next event
    float bsdfDirPdfW;
    float cosThetaOut;
    float3 bsdfSample = getRandomUniformFloat3(&aCameraPrd.randomState);
    float3 bsdfFactor = cameraBsdf.vcmSampleF(&aCameraPrd.direction, bsdfSample, &bsdfDirPdfW, &cosThetaOut); // CUDA 6 fails here
    //OPTIX_PRINTFI(aCameraPrd.depth, "Hit - new dir %f %f %f\n", aCameraPrd.direction.x, aCameraPrd.direction.y, aCameraPrd.direction.z);

    float bsdfRevPdfW = cosThetaIn * M_1_PIf;
    bsdfDirPdfW *= contProb;
    bsdfRevPdfW *= contProb;
    updateMisTermsOnScatter(aCameraPrd, cosThetaOut, bsdfDirPdfW, bsdfRevPdfW, aMisVcWeightFactor, aMisVmWeightFactor, pVertexPickPdf);

    // f * cosTheta / f_pdf
    aCameraPrd.throughput *= bsdfFactor * (cosThetaOut / bsdfDirPdfW);
    aCameraPrd.origin = aHitPoint;
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "Hit C - new origin     % 14f % 14f % 14f\n", 
        aCameraPrd.origin.x, aCameraPrd.origin.y, aCameraPrd.origin.z);
}
