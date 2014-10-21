/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Valdis Vilcans
*/

#pragma once

#define OPTIX_PRINTF_DEF
#define OPTIX_PRINTFI_DEF
#define OPTIX_PRINTFID_DEF
#define OPTIX_PRINTFC_DEF
#define OPTIX_PRINTFCID_DEF

#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "renderer/helpers/helpers.h"
#include "renderer/RayType.h"
#include "renderer/ShadowPRD.h"
#include "renderer/helpers/samplers.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/light.h"
#include "renderer/Light.h"
#include "renderer/Camera.h"
#include "renderer/BSDF.h"
#include "renderer/vcm/LightVertex.h"
#include "renderer/vcm/SubpathPRD.h"
#include "renderer/vcm/config_vcm.h"
#include "renderer/vcm/mis.h"

//#define CONNECT_VERTICES_DISABLED
//#define CONNECT_CAMERA_T1_DISABLED
//#define CONNECT_LIGHT_S0_DISABLED
//#define CONNECT_LIGHT_S1_DISABLED

#define OPTIX_PRINTF_ENABLED 0
#define OPTIX_PRINTFI_ENABLED 0
#define OPTIX_PRINTFID_ENABLED 0

RT_FUNCTION int isOccluded( const rtObject      & aSceneRootObject, 
                            const optix::float3 & aPoint, 
                            const optix::float3 & aDirection, 
                            const float           aTMax )
{
    using namespace optix;
    if (aTMax < 3.f*EPS_RAY)
        return false;

    ShadowPRD shadowPrd;
    shadowPrd.attenuation = 1.0f;
    Ray occlusionRay(aPoint, aDirection, RayType::SHADOW, EPS_RAY, aTMax - 2.f*EPS_RAY);

    rtTrace(aSceneRootObject, occlusionRay, shadowPrd);
    return shadowPrd.attenuation == 0.f;
}



#define OPTIX_PRINTFC_ENABLED 0
#define OPTIX_PRINTFID_ENABLED 0
#define OPTIX_PRINTFCID_ENABLED 0
RT_FUNCTION void connectCameraT1( const rtObject        & aSceneRootObject,
                                  SubpathPRD            & aLightPrd,
                                  const VcmBSDF         & aLightBsdf,
                                  const optix::float3   & aLightHitpoint,
                                  const optix::uint     & aLightSubpathCount,
                                  const float             aMisVmWeightFactor,
                                  const Camera          & aCamera,
                                  const optix::float2   & aPixelSizeFactor,
                                  rtBufferId<float3, 2>   aOutputBuffer,
                                  optix::uint2          * oConnectedPixel = NULL )
{
    using namespace optix;
        
    // Check if point in front of camera
    float3 dirToCamera = aCamera.eye - aLightHitpoint;
    if (dot(aCamera.lookdir, -dirToCamera) <= 0.f)
        return;

    const float distance = length(dirToCamera);
    dirToCamera /= distance;
    const float cosAtCamera = dot(normalize(aCamera.lookdir), -dirToCamera);

    // Get image plane pos
    const float imagePointToCameraDist = length(aCamera.lookdir) / cosAtCamera;
    const float3 imagePlanePointWorld = aCamera.eye + (-dirToCamera * imagePointToCameraDist);

    const float3 imagePlaneCenterWorld = aCamera.eye + aCamera.lookdir;
    const float3 imagePlaneCenterToPointWorld = imagePlanePointWorld - imagePlaneCenterWorld;

    // Check if is within image coords
    const float3 unitCameraU = normalize(aCamera.camera_u);
    const float3 unitCameraV = normalize(aCamera.camera_v);
    const float proj_u_len = dot(imagePlaneCenterToPointWorld, unitCameraU);
    const float proj_v_len = dot(imagePlaneCenterToPointWorld, unitCameraV);
    
    float2 posOnPlane = make_float2(proj_u_len, proj_v_len); // image plane pos relative to center
    int isOnImage = 2.f * fabs(posOnPlane.x) < aCamera.imagePlaneSize.x && 2.f * fabs(proj_v_len) < aCamera.imagePlaneSize.y;

    if (!isOnImage)
        return;

    // Get pixel index
    float2 pixelCoord = (posOnPlane + 0.5f * aCamera.imagePlaneSize) / aCamera.imagePlaneSize;
    size_t2 screenSize = aOutputBuffer.size();
    uint2 pixelIndex = make_uint2(pixelCoord.x * screenSize.x, pixelCoord.y * screenSize.y);
    pixelIndex.x = clamp(pixelIndex.x, 0u, screenSize.x-1);
    pixelIndex.y = clamp(pixelIndex.y, 0u, screenSize.y-1);
    
    // get bsdf factor and dir/rev pdfs
    float cosToCamera, bsdfDirPdfW, bsdfRevPdfW;
    const float3 bsdfFactor = aLightBsdf.vcmF(dirToCamera, cosToCamera, &bsdfDirPdfW, &bsdfRevPdfW, &aLightPrd.launchIndex);

    if (isZero(bsdfFactor))
        return;

    bsdfRevPdfW *= aLightBsdf.continuationProb();

    // Conversion factor from image plane area to surface area
    const float imageToSolidAngleFactor = sqr(imagePointToCameraDist) / cosAtCamera;

    // Image plane is sampled per pixel when generating rays, so use pixel are pdf
    const float pixelArea = aPixelSizeFactor.x * aCamera.imagePlaneSize.x * aPixelSizeFactor.x * aCamera.imagePlaneSize.y;
    float imageSamplePdfA = 1.f / pixelArea;

    // pdf factors computed step by step and labeled as in [tech. rep. (46)]
    const float cameraPdfW = imageSamplePdfA * imageToSolidAngleFactor;      // p_ro_1
    const float cameraPdfA = cameraPdfW * fabs(cosToCamera) / sqr(distance); // p1 = p_ro_1 * g1
    const float lightHitpointRevPdfA = cameraPdfA;                           // _p_s-1 - pdf for sampling aHitpoint as part of camera path

    // Partial light sub-path weight [tech. rep. (46)]. Note the division by aLightSubpathCount, which is the number 
    // of samples this technique uses (e.g. all light subpaths try to connect to camera at every hitpoint).
    // This division also appears a few lines below in the frame buffer accumulation.
    //
    // wLight also needs to account for different image point sampling techniques that could be used when 
    // generating (p0trace) or when connecting to camera (p0connect). In our case both p0connect and p0trace 
    // are imageSamplePdfA and cancel out
    const float wLight = vcmMis(lightHitpointRevPdfA / aLightSubpathCount) * // * p0connect/p0trace *
        (aMisVmWeightFactor + aLightPrd.dVCM + aLightPrd.dVC * vcmMis(bsdfRevPdfW));

    // Partial eye sub-path weight is 0 [tech. rep. (47)]

    // Full path MIS weight [tech. rep. (37)]. No MIS for traditional light tracing.
    const float misWeight = 1.f / (wLight + 1.f);

    // Pixel integral is over image plane area, hence we need to convert invert cameraPdfA to get
    // image area pdf. cameraPdfA is imageSamplePdf converted to represent possibility to sample point aLightHitpoint
    // so we invert it to convert back to image area pdf
    const float cameraPdfAConvertedToImagePdfA = 1.f / cameraPdfA;

    // We also divide by the number of samples this technique makes, which is equal to the number of light sub-paths
    float3 contrib = misWeight * aLightPrd.throughput * bsdfFactor / (aLightSubpathCount * cameraPdfAConvertedToImagePdfA);

    if (!isOccluded(aSceneRootObject, aLightHitpoint, dirToCamera, distance))
    {
        aOutputBuffer[pixelIndex] += contrib;
    }
}



#define OPTIX_PRINTFID_ENABLED 0
RT_FUNCTION void sampleScattering( SubpathPRD                   & aSubpathPrd,
                                   const optix::float3          & aHitPoint, 
                                   const VcmBSDF                & aBsdf,
                                   const float                    aMisVcWeightFactor,
                                   const float                    aMisVmWeightFactor )
{
    // Russian Roulette
    float contProb = aBsdf.continuationProb();
    float rrSample = getRandomUniformFloat(&aSubpathPrd.randomState);    
    if (contProb < rrSample)
    {
        aSubpathPrd.done = true;
        return;
    }

    //next event
    float bsdfDirPdfW = 0.f;
    float cosThetaOut = 0.f;
    float3 bsdfSample = getRandomUniformFloat3(&aSubpathPrd.randomState);
    BxDF::Type sampledEvent;
    float3 bsdfFactor = aBsdf.vcmSampleF(&aSubpathPrd.direction, bsdfSample, &bsdfDirPdfW, &cosThetaOut, BxDF::All, &sampledEvent); // CUDA 6 fails here
    
    if (isZero(bsdfFactor))
        return;

    float bsdfRevPdfW = bsdfDirPdfW;
    bool isSpecularEvent = BxDF::matchFlags(sampledEvent, BxDF::Specular);
    if (!isSpecularEvent)       // evaluate pdf for non-specular event, otherwise it is the same as direct pdf
        bsdfRevPdfW = aBsdf.pdf(aSubpathPrd.direction, BxDF::Type(BxDF::All & ~BxDF::Specular), true);
    aSubpathPrd.isSpecularPath = (aSubpathPrd.isSpecularPath && isSpecularEvent);
    bsdfDirPdfW *= contProb;
    bsdfRevPdfW *= contProb;

    updateMisTermsOnScatter(aSubpathPrd, cosThetaOut, bsdfDirPdfW, bsdfRevPdfW, aMisVcWeightFactor, aMisVmWeightFactor, sampledEvent /*, pVertexPickPdf*/);

    // f * cosTheta / f_pdf
    aSubpathPrd.throughput *= bsdfFactor * (cosThetaOut / bsdfDirPdfW); 
    aSubpathPrd.origin = aHitPoint;
}



#define OPTIX_PRINTFID_ENABLED 0
#define OPTIX_PRINTFCID_ENABLED 0
RT_FUNCTION void lightHit( const rtObject               & aSceneRootObject,
                           SubpathPRD                   & aLightPrd,
                           const optix::float3          & aHitPoint, 
                           const optix::float3          & aWorldNormal,
                           const VcmBSDF                & aLightBsdf,
                           const optix::float3            aRayWorldDir,  // not passing ray dir by reference sine it's OptiX semantic type
                           const float                    aRayTHit,
                           const optix::uint              aMaxPathLen,
                           const optix::uint              aLightVertexCountEstimatePass,
                           const optix::uint              aLightSubpathCount,
                           const float                    aMisVcWeightFactor,
                           const float                    aMisVmWeightFactor,
                           const Camera                 & aCamera,
                           const optix::float2            aPixelSizeFactor,
                           rtBufferId<float3, 2>          aOutputBuffer,
                           rtBufferId<LightVertex, 1>     aLightVertexBuffer,
                           rtBufferId<optix::uint, 1>     aLightVertexBufferIndexBuffer,
                           rtBufferId<optix::uint, 1>     aLightSubpathVertexCountBuffer,
#if !VCM_UNIFORM_VERTEX_SAMPLING                         // for 1 to 1 camera - light path connections
                           rtBufferId<optix::uint, 2>     aLightSubpathVertexIndexBuffer
#else                                                    // uniform vertex sampling
                           const float                  * aVertexPickPdf
#endif
                           )
{
    using namespace optix;

#if VCM_UNIFORM_VERTEX_SAMPLING
    const float *pVertexPickPdf = aVertexPickPdf;
#else
    const float *pVertexPickPdf = NULL;
#endif

    aLightPrd.depth++;

    // vmarz TODO infinite lights need additional handling
    float cosThetaIn = dot(aWorldNormal, -aRayWorldDir);
    if (cosThetaIn < EPS_COSINE) // reject if cos too low
    {
        aLightPrd.done = true;
        return;
    }   

    updateMisTermsOnHit(aLightPrd, cosThetaIn, aRayTHit);

    bool isBsdfSpecular = aLightBsdf.isSpecular();
    if (!isBsdfSpecular)
    {
        // vertex count can be lower that path length since not stored on specular surfaces
        uint currPathVertIdx = aLightSubpathVertexCountBuffer[aLightPrd.launchIndex1D]++;
        
        // store path vertex
        if (!aLightVertexCountEstimatePass)
        {
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
            lightVertex.bsdf = aLightBsdf;

            // Store in buffer
            uint vertIdx = atomicAdd(&aLightVertexBufferIndexBuffer[0], 1u);
            aLightVertexBuffer[vertIdx] = lightVertex;

#if !VCM_UNIFORM_VERTEX_SAMPLING
            //uint3 pathVertIdx = make_uint3(launchIndex, aLightPrd.depth-1); // getting this ?? 1072693248 0 0 or 1072693248 1 0
            uint2 pathVertIdx = make_uint2(aLightPrd.launchIndex1D, currPathVertIdx);  // can't use depth for index since not storing vertex at every hit
            aLightSubpathVertexIndexBuffer[pathVertIdx] = vertIdx;
#endif
        }
    }


#ifndef CONNECT_CAMERA_T1_DISABLED
    if (!aLightVertexCountEstimatePass && !isBsdfSpecular)
    {
        connectCameraT1(aSceneRootObject, aLightPrd, aLightBsdf, aHitPoint, aLightSubpathCount,
                        aMisVmWeightFactor, aCamera, aPixelSizeFactor, aOutputBuffer);
    }
#endif

    // Terminate if path would become too long after scattering (e.g. +2 for next hit and connection)
    if (aMaxPathLen < aLightPrd.depth + 2)
    {
        aLightPrd.done = true;
        return;
    }

    sampleScattering(aLightPrd, aHitPoint, aLightBsdf, aMisVcWeightFactor, aMisVmWeightFactor);
}



#define OPTIX_PRINTFID_ENABLED 0
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

    // Get connection
    float3 direction = aLightVertex.hitPoint - aCameraHitpoint;
    float dist2      = dot(direction, direction);
    float distance   = sqrt(dist2);
    direction       /= distance;

    // Evaluate BSDF at camera vertex
    float cameraCosTheta, cameraBsdfDirPdfW, cameraBsdfRevPdfW;
    const float3 cameraBsdfFactor = aCameraBsdf.vcmF(direction, cameraCosTheta, &cameraBsdfDirPdfW, &cameraBsdfRevPdfW);

    if (isZero(cameraBsdfFactor))
        return;

    // Add camera continuation probability (for russian roulette)
    const float cameraCont = aCameraBsdf.continuationProb();
    cameraBsdfDirPdfW *= cameraCont;
    cameraBsdfRevPdfW *= cameraCont;

    // Evaluate BSDF at light vertex
    float lightCosTheta, lightBsdfDirPdfW, lightBsdfRevPdfW;
    const float3 lightBsdfFactor = aLightVertex.bsdf.vcmF(-direction, lightCosTheta, &lightBsdfDirPdfW, &lightBsdfRevPdfW);
    
    if (isZero(lightBsdfFactor))
        return;

    // Add camera continuation probability (for russian roulette)
    const float lightCont = aLightVertex.bsdf.continuationProb();
    lightBsdfDirPdfW *= lightCont;
    lightBsdfRevPdfW *= lightCont;

    // Geometry term
    const float geometryTerm = lightCosTheta * cameraCosTheta / dist2;
    if (geometryTerm < 0.f)
        return;

    // Convert solid angle pdfs to area pdfs
    const float cameraBsdfDirPdfA = pdfWtoA(cameraBsdfDirPdfW, distance, cameraCosTheta);
    const float lightBsdfDirPdfA = pdfWtoA(lightBsdfDirPdfW, distance, lightCosTheta);

    // aVertPickPdf is set only when unform vertex sampling used (connecting to all paths)
    float invVertPickPdf = aVertexPickPdf ? (1.f / *aVertexPickPdf) : 1.f;
    float aCameraPrd_dVC = aCameraPrd.dVC;

#if VCM_UNIFORM_VERTEX_SAMPLING
    aCameraPrd_dVC = aCameraPrd.dVC_unif_vert;
    // There is no dVC_unif_vert in LightVertex since vertices are used only for connection between each other,
    // and do not affect connection to camera/light source and dVC is not present in weight equation for VM.
    // equations in [tech. rep. (38-47)]
    OPTIX_PRINTFID(aCameraPrd.launchIndex, aCameraPrd.depth, "conn  -  invVertPickPdf % 14f \n", invVertPickPdf);
#endif

    // Partial light sub-path MIS weight [tech. rep. (40)]
    const float wLight = vcmMis(cameraBsdfDirPdfA) * 
        ( aMisVmWeightFactor * invVertPickPdf + aLightVertex.dVCM + aLightVertex.dVC * vcmMis(lightBsdfRevPdfW) );
    // lightBsdfRevPdfW is Reverse with respect to light path, e.g. in eye path progression 
    // dirrection (note same arrow dirs in formula)
    // note (40) and (41) uses light subpath Y and camera subpath z;

    // Partial eye sub-path MIS weight [tech. rep. (41)]
    const float wCamera = vcmMis(lightBsdfDirPdfA) * 
        ( aMisVmWeightFactor * invVertPickPdf + aCameraPrd.dVCM + aCameraPrd_dVC * vcmMis(cameraBsdfRevPdfW) );

    // Full path MIS weight [tech. rep. (37)]
    const float misWeight = 1.f / (wLight + 1.f + wCamera);

    float3 contrib = geometryTerm * cameraBsdfFactor * lightBsdfFactor * invVertPickPdf;
    contrib *= misWeight * aCameraPrd.throughput * aLightVertex.throughput;

    // TODO try early occlusion check
    if (isOccluded(aSceneRootObject, aCameraHitpoint, direction, distance))
        return;

    aCameraPrd.color += contrib;
}


#define OPTIX_PRINTFID_ENABLED 0
// Connects camera subpath vertex to light source, e.g. direct illumination, next event estimation.
// Light subpath length=1 [tech. rep 44-45]
RT_FUNCTION void connectLightSourceS1( const rtObject             & aSceneRootObject,
                                       const Sphere               & aSceneBoundingSphere,
                                       const rtBufferId<Light, 1>   aLightsBuffer,
                                       SubpathPRD                 & aCameraPrd,
                                       const VcmBSDF              & aCameraBsdf,
                                       const optix::float3        & aCameraHitpoint,
                                       const float                  aMisVmWeightFactor,
                                       const float const          * aVertexPickPdf = NULL)
{
    using namespace optix;

    int lightIndex = 0;
    if (1 < aLightsBuffer.size())
    {
        float sample = getRandomUniformFloat(&aCameraPrd.randomState);
        lightIndex = intmin((int)(sample*aLightsBuffer.size()), int(aLightsBuffer.size()-1));
    }

    const Light light               = aLightsBuffer[lightIndex];
    const float inverseLightPickPdf = aLightsBuffer.size();
    const float lightPickProb        = 1.f / aLightsBuffer.size();

    float emissionPdfW;
    float directPdfW;
    float cosAtLight;
    float distance;
    float3 dirToLight;
    float3 radiance = lightIlluminate(aSceneBoundingSphere, light, aCameraPrd.randomState, aCameraHitpoint, dirToLight,
        distance, directPdfW, &emissionPdfW, &cosAtLight);
    
    if (isZero(radiance))
        return;

    float bsdfDirPdfW, bsdfRevPdfW, cosToLight;
    float3 bsdfFactor = aCameraBsdf.vcmF(dirToLight, cosToLight, &bsdfDirPdfW, &bsdfRevPdfW, &aCameraPrd.launchIndex);

    if (isZero(bsdfFactor))
        return;

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

    // Comment from SmallVCM
    // Partial eye sub-path MIS weight [tech. rep. (45)].
    //
    // In front of the sum in the parenthesis we have vcmMis(ratio), where
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
    float3 contrib = (misWeight * cosToLight / (lightPickProb * directPdfW)) * (radiance * bsdfFactor);

    if (isZero(contrib))
        return;

    if (isOccluded(aSceneRootObject, aCameraHitpoint, dirToLight, distance))
        return;

    aCameraPrd.color += contrib * aCameraPrd.throughput;
}


// Computes vertex connection contribution for camera path vertex that has been sampled on 
// light source (traced and randomly hit the light). Light subpath length = 0 [tech. rep 42-43]
RT_FUNCTION void connectLightSourceS0(SubpathPRD & aCameraPrd, const optix::float3 &aRadiance, float & aDirectPdfA, 
                                     float & aEmissionPdfW, float aLightPickProb, int aUseVC, int aUseVM)
{
    if (aCameraPrd.depth == 1) // first hit, seen directly from camera, no weighting needed
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

    aDirectPdfA *= aLightPickProb;  // p0connect in [tech. rep. (43)].
    aEmissionPdfW *= aLightPickProb;// p0trace in [tech. rep. (43)].

    // Partial eye sub-path MIS weight [tech. rep. (43)].
    // If the last hit was specular, then dVCM == 0.
    const float wCamera = vcmMis(aDirectPdfA) * aCameraPrd.dVCM + vcmMis(aEmissionPdfW * aCameraPrd.dVC);
    // Partial light sub-path weight is 0 [tech. rep. (42)].

    // Full path MIS weight [tech. rep. (37)].
    const float misWeight = 1.f / (1.f + wCamera);

    aCameraPrd.color += aCameraPrd.throughput * misWeight * aRadiance;
}



#define OPTIX_PRINTFID_ENABLED 0
RT_FUNCTION void cameraHit( const rtObject                     & aSceneRootObject,
                            const Sphere                       & aSceneBoundingSphere,
                            SubpathPRD                         & aCameraPrd,
                            const optix::float3                & aHitPoint,
                            const optix::float3                & aWorldNormal,
                            const VcmBSDF                      & aCameraBsdf,
                            const optix::float3                  aRayWorldDir,  // not passing ray dir by reference since it's OptiX semantic type
                            const float                          aRayTHit,
                            const optix::uint                    aMaxPathLen,
                            const optix::uint                    aLightSubpathCount,
                            const float                          aMisVcWeightFactor,
                            const float                          aMisVmWeightFactor,
                            const rtBufferId<Light, 1>           aLightsBuffer,
                            const rtBufferId<LightVertex>        aLightVertexBuffer,
                            const rtBufferId<optix::uint>        aLightVertexBufferIndexBuffer,
                            const rtBufferId<optix::uint>        aLightSubpathVertexCountBuffer,
#if !VCM_UNIFORM_VERTEX_SAMPLING                                // for 1 to 1 camera - light path connections
                            const rtBufferId<optix::uint, 2>     aLightSubpathVertexIndexBuffer
#else                                                           // uniform vertex sampling
                            const float                          averageLightSubpathLength,
                            const float                        * aVertexPickPdf
#endif
                            )
{
    using namespace optix;

#if VCM_UNIFORM_VERTEX_SAMPLING
    const float *pVertexPickPdf = aVertexPickPdf;
#else
    const float *pVertexPickPdf = NULL;
#endif

    aCameraPrd.depth++;	

    // vmarz TODO infinite lights need additional handling
    float cosThetaIn = dot(aWorldNormal, -aRayWorldDir);
    if (cosThetaIn < EPS_COSINE) // reject if cos too low
    {
        aCameraPrd.done = true;
        return;
    }   

    updateMisTermsOnHit(aCameraPrd, cosThetaIn, aRayTHit);
    
    bool isBsdfSpecular = aCameraBsdf.isSpecular();
#ifndef CONNECT_LIGHT_S1_DISABLED
    if (!isBsdfSpecular)
    {
        // Connect by sampling a vetex on light source, e.g. light path length = 1
        connectLightSourceS1(aSceneRootObject, aSceneBoundingSphere, aLightsBuffer, aCameraPrd, aCameraBsdf, aHitPoint, aMisVmWeightFactor);
    }
#endif
    
    // Connect to light vertices
    if (!isBsdfSpecular)
    {
#ifndef CONNECT_VERTICES_DISABLED
#if VCM_UNIFORM_VERTEX_SAMPLING
        uint numLightVertices = aLightVertexBufferIndexBuffer[0];
        //float vertexPickPdf = float(vcmNumlightVertexConnections) / numLightVertices; // TODO scale by pick prob
        uint numlightVertexConnections = ceilf(averageLightSubpathLength);
        float lastVertConnectProb = averageLightSubpathLength - (uint)averageLightSubpathLength;
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
        // CAUTION: this loop can cause weird issues, out of bound access with crazy indices (though they are based 
        // on launch index and loop variable), rtTrace crashing within the loop etc.
        // update: It seems it was caused multiple uses of std::printf
        uint pathIndex = aCameraPrd.launchIndex1D % aLightSubpathCount;
        uint lightSubpathVertexCount = aLightSubpathVertexCountBuffer[pathIndex]; // vertex count can be lower that path length since not stored on specular surfaces
        for (uint i = 0; i < lightSubpathVertexCount; ++i)
        {
            uint2 pathVertIdx = make_uint2(pathIndex, i);
            uint vertIdx = aLightSubpathVertexIndexBuffer[pathVertIdx];
            LightVertex lightVertex = aLightVertexBuffer[vertIdx];
            connectVertices(aSceneRootObject, lightVertex, aCameraPrd, aCameraBsdf, aHitPoint, aMisVmWeightFactor);
        }
#endif
#endif
    }

    // Terminate if path too long for connections and merging
    if (aMaxPathLen <= aCameraPrd.depth)
    {
        aCameraPrd.done = true;
        return;
    }

    sampleScattering(aCameraPrd, aHitPoint, aCameraBsdf, aMisVcWeightFactor, aMisVmWeightFactor);
}

#undef OPTIX_PRINTF_ENABLED
#undef OPTIX_PRINTFI_ENABLED
#undef OPTIX_PRINTFID_ENABLED
#undef OPTIX_PRINTFC_ENABLED
#undef OPTIX_PRINTFCID_ENABLED