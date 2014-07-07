#pragma once
#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "renderer/helpers/samplers.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/light.h"
#include "renderer/Light.h"
#include "renderer/Camera.h"
#include "renderer/helpers/helpers.h"
#include "renderer/vcm/SubpathPRD.h"
#include "renderer/vcm/config_vcm.h"


// Initialize light payload - throughput premultiplied with light radiance, partial MIS terms  [tech. rep. (31)-(33)]
__inline__ __device__ void initLightPayload(SubpathPRD & aLightPrd, const Light & aLight, const float & aLightPickPdf,
                                          const float & misVcWeightFactor, const float const * aVertexPickPdf = NULL)
{
    using namespace optix;

    float emissionPdfW;
    float directPdfW;
    float cosAtLight;
    aLightPrd.throughput = lightEmit(aLight, aLightPrd.randomState, aLightPrd.origin, aLightPrd.direction,
        emissionPdfW, directPdfW, cosAtLight, &aLightPrd.launchIndex);
    // vmarz?: do something similar as done for photon emission, emit towards scene when light far from scene?

    emissionPdfW *= aLightPickPdf;
    directPdfW *= aLightPickPdf;
    aLightPrd.throughput /= emissionPdfW;
    //lightPrd.isFinite = isDelta.isFinite ... vmarz?
    OPTIX_PRINTFID(aLightPrd.launchIndex, "GenLi - emission Pdf    % 14f     directPdfW % 14f\n", 
        emissionPdfW, directPdfW);
    OPTIX_PRINTFID(aLightPrd.launchIndex, "GenLi - prd throughput  % 14f % 14f % 14f\n", 
        aLightPrd.throughput .x, aLightPrd.throughput .y, aLightPrd.throughput .z);

    // Partial light sub-path MIS quantities. [tech. rep. (31)-(33)]
    // Evaluation is completed after tracing the emission ray
    aLightPrd.dVCM = vcmMis(directPdfW / emissionPdfW);
    // dVCM_1 = p0_connect / ( p0_trace * p1 )
    //    connect/trace refer to potentially potentially different techniques for sampling points depending if point is 
    //    used to connect to a subpath or as a starting point of a new one
    // directPdfW = p0_connect = areaSamplePdfA * lightPickPdf
    // emissionPdfW = p0_trace * p1
    //    p0_trace = areaSamplePdf * lightPickPdf
    //    p1 = directionSamplePdfW * g1 = (cos / Pi) * g1 [g1 added after tracing]

    // e.g. if not delta light
    if (!aLight.isDelta)
    {
    	const float usedCosLight = aLight.isFinite ? cosAtLight : 1.f;
    	aLightPrd.dVC = vcmMis(usedCosLight / emissionPdfW);
        // dVC_1 = _g0 / ( p0_trace * p1 ) 
        // usedCosLight is part of _g0 - reverse pdf conversion factor!, uses outgoing cos not incident at next vertex,
        //    sqr(dist) from _g0 added after tracing
        // emissionPdfW = p0_trace * p1

#if VCM_UNIFORM_VERTEX_SAMPLING
        aLightPrd.dVC_unif_vert = aLightPrd.dVC;
#endif
    }

    // dVM_1 = dVC_1 / etaVCM
    aLightPrd.dVM = aLightPrd.dVC * misVcWeightFactor;
    if (aVertexPickPdf)
        aLightPrd.dVM *= *aVertexPickPdf; // should divide etaVCM, bust since it is divider, we just multiply
                                          // aVertexPickPdf pointer passed only when using uniform vertex sampling from buffer
}


// Initialize camera payload - partial MIS terms [tech. rep. (31)-(33)]
__inline__ __device__ void initCameraPayload(SubpathPRD & aCameraPrd, const Camera & aCamera, 
                                           const optix::float2 & aPixelSizeFactor, const optix::uint & aVcmLightSubpathCount)
{
    using namespace optix;

    // pdf conversion factor from area on image plane to solid angle on ray
    float cosAtCamera = dot(normalize(aCamera.lookdir), aCameraPrd.direction);
    float distToImgPlane = length(aCamera.lookdir);
    float imagePointToCameraDist = length(aCamera.lookdir) / cosAtCamera;
    float imageToSolidAngleFactor = sqr(imagePointToCameraDist) / cosAtCamera;

    float pixelArea = aPixelSizeFactor.x * aCamera.imagePlaneSize.x * aPixelSizeFactor.x * aCamera.imagePlaneSize.y;
    float areaSamplePdf = 1.f / pixelArea;

    // Needed if use different image point sampling techniques, see p0connect/p0trace in dVCM comment below
    //float p0connect = areaSamplePdf;      // cancel out
    //float p0trace = areaSamplePdf;        // cancel out
    float cameraPdf = areaSamplePdf * imageToSolidAngleFactor;
    //OPTIX_PRINTFID(aCameraPrd.launchIndex, "Gen C - init  - cosC %f planeDist %f pixA solidAngleFact %f camPdf %f\n", 
    //    cosAtCamera, distToImgPlane, imageToSolidAngleFactor, pixelArea);

    // Initialize sub-path MIS quantities, partially [tech. rep. (31)-(33)]
    aCameraPrd.dVC = .0f;
    aCameraPrd.dVM = .0f;
#if VCM_UNIFORM_VERTEX_SAMPLING
    aCameraPrd.dVC_unif_vert = aCameraPrd.dVC;
#endif

    // dVCM = ( p0connect / p0trace ) * ( nLightSamples / p1 )
    // p0connect/p0trace - potentially different sampling techniques 
    //      p0connect - pdf for technique used when connecting to camera  during light tracing step
    //      p0trace - pdf for technique used when sampling a ray starting point
    // p1 = p1_ro * g1 = areaSamplePdf * imageToSolidAngleFactor * g1 [g1 added after tracing]
    // p0connect/p0trace cancel out in our case
    aCameraPrd.dVCM = vcmMis( aVcmLightSubpathCount / cameraPdf );
    //OPTIX_PRINTFID(aCameraPrd.launchIndex, "Gen C - init  - dVCM %f lightSubCount %d camPdf %f\n", aCameraPrd.dVCM, 
    //    aVcmLightSubpathCount, cameraPdf);

    //cameraPrd.specularPath = 1; // vmarz TODO ?
}



// Update MIS quantities before storing at the vertex, follows initialization on light [tech. rep. (31)-(33)]
// or scatter from surface [tech. rep. (34)-(36)]
__inline__ __device__ void updateMisTermsOnHit(SubpathPRD & aLightPrd, const float & aCosThetaIn, const float & aRayLen)
{
    // sqr(dist) term from g in 1/p1 (or 1/pi), for dVC and dVM sqr(dist) terms of _g and pi cancel out
    aLightPrd.dVCM *= vcmMis(sqr(aRayLen));
    aLightPrd.dVCM /= vcmMis(aCosThetaIn);
    aLightPrd.dVC  /= vcmMis(aCosThetaIn);
    aLightPrd.dVM  /= vcmMis(aCosThetaIn);
#if VCM_UNIFORM_VERTEX_SAMPLING
    aLightPrd.dVC_unif_vert  /= vcmMis(aCosThetaIn);
#endif
}





// Initializes MIS terms for next event, partial implementation of [tech. rep. (34)-(36)], completed on hit
__inline__ __device__ void updateMisTermsOnScatter(SubpathPRD & aPathPrd, const float & aCosThetaOut, const float & aBsdfDirPdfW,
                                                 const float & aBsdfRevPdfW, const float & aMisVcWeightFactor, const float & aMisVmWeightFactor,
                                                 const float const * aVertexPickPdf = NULL)
{
    OPTIX_PRINTFID(aPathPrd.launchIndex, "updateMisTermsOnScatter(): \n");
    float vertPickPdf = aVertexPickPdf ?  (*aVertexPickPdf) : 1.f;
    const float dVC = aPathPrd.dVC;
    const float dVM = aPathPrd.dVM;
    const float dVCM = aPathPrd.dVCM;
    OPTIX_PRINTFID(aPathPrd.launchIndex, "MIS   -            dVC % 14f            dVM % 14f           dVCM % 14f \n", dVC, dVM, dVCM);

    // dVC = (g_i-1 / pi) * (etaVCM + dVCM_i-1 + _p_ro_i-2 * dVC_i-1)
    // cosThetaOut part of g_i-1  [ _g reverse pdf conversion!, uses outgoing cosTheta]
    //   !! sqr(dist) terms for _g_i-1 and gi of pi are the same and cancel out, hence NOT scaled after tracing]
    // pi = bsdfDirPdfW * g1
    // bsdfDirPdfW = _p_ro_i    [part of pi]
    // bsdfRevPdfW = _p_ro_i-2
    aPathPrd.dVC = vcmMis(aCosThetaOut / aBsdfDirPdfW) * ( 
        aPathPrd.dVC * vcmMis(aBsdfRevPdfW) +              
        aPathPrd.dVCM + aMisVmWeightFactor);               

    OPTIX_PRINTFID(aPathPrd.launchIndex,
        "MIS   -          U dVC = (   cosThetaOut /    bsdfDirPdfW) * (           dVC *    bsdfRevPdfW +           dVCM + VmWeightFactor) \n");
    OPTIX_PRINTFID(aPathPrd.launchIndex, "MIS   - % 14f = (% 14f / % 14f) * (% 14e * % 14f + % 14e + % 14f) \n", 
        aPathPrd.dVC, aCosThetaOut, aBsdfDirPdfW, dVC, aBsdfRevPdfW, dVCM, aMisVmWeightFactor);

#if VCM_UNIFORM_VERTEX_SAMPLING
    const float dVC_unif_vert = aPathPrd.dVC_unif_vert;
    aPathPrd.dVC_unif_vert = vcmMis(aCosThetaOut / aBsdfDirPdfW) * ( 
        aPathPrd.dVC_unif_vert * vcmMis(aBsdfRevPdfW) +              
        aPathPrd.dVCM + aMisVmWeightFactor / vertPickPdf);
    OPTIX_PRINTFID(aPathPrd.launchIndex,
        "MIS   - U dVC_unifvert = (   cosThetaOut /    bsdfDirPdfW) * ( dVC_unif_vert *    bsdfRevPdfW +           dVCM + VmWeightFactor /    vertPickPdf) \n");
    OPTIX_PRINTFID(aPathPrd.launchIndex, "MIS   - % 14f = (% 14f / % 14f) * (% 14e * % 14f + % 14e + % 14f / % 14e) \n", 
        aPathPrd.dVC_unif_vert, aCosThetaOut, aBsdfDirPdfW, dVC_unif_vert, aBsdfRevPdfW, dVCM, aMisVmWeightFactor, vertPickPdf);
#endif

    // dVM = (g_i-1 / pi) * (1 + dVCM_i-1/etaVCM + _p_ro_i-2 * dVM_i-1)
    // cosThetaOut part of g_i-1 [_g reverse pdf conversion!, uses outgoing cosTheta]
    //    !! sqr(dist) terms for _g_i-1 and gi of pi are the same and cancel out, hence NOT scaled after tracing]
    aPathPrd.dVM = vcmMis(aCosThetaOut / aBsdfDirPdfW) * ( 
        aPathPrd.dVM * vcmMis(aBsdfRevPdfW) +              
        aPathPrd.dVCM * aMisVcWeightFactor * vertPickPdf + 1.f ); // vertPickPdf should divide etaVCM which is inverse in aMisVcWeightFactor
    OPTIX_PRINTFID(aPathPrd.launchIndex, "MIS   -          U dVM = (   cosThetaOut /    bsdfDirPdfW) * (           dVM *    bsdfRevPdfW +           dVCM + VcWeightFactor *    vertPickPdf + 1) \n");
    OPTIX_PRINTFID(aPathPrd.launchIndex, "MIS   - % 14f = (% 14f / % 14f) * (% 14e * % 14f + % 14e + % 14f * % 14f + 1) \n", 
        aPathPrd.dVM, aCosThetaOut, aBsdfDirPdfW, dVM, aBsdfRevPdfW, dVCM, aMisVcWeightFactor, vertPickPdf);

    // dVCM = 1 / pi
    // pi = bsdfDirPdfW * g1 = _p_ro_i * g1 [only for dVCM sqe(dist) terms do not cancel out and are added after tracing]
    aPathPrd.dVCM = vcmMis(1.f / aBsdfDirPdfW);
    OPTIX_PRINTFID(aPathPrd.launchIndex, "MIS   -         U dVCM = (1 /    bsdfDirPdfW) \n");
    OPTIX_PRINTFID(aPathPrd.launchIndex, "MIS   - % 14f = (1 / %14f) \n",  dVCM, aBsdfDirPdfW);
    OPTIX_PRINTFID(aPathPrd.launchIndex, "MIS   -          U dVC % 14f          U dVM % 14f         U dVCM % 14f \n", aPathPrd.dVC, aPathPrd.dVM, aPathPrd.dVCM);
}