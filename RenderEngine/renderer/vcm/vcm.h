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


// Initialize light payload - througput premultiplied with light radiance, partial MIS terms  [tech. rep. (31)-(33)]
optix::float3 __inline __device__ initLightPayload(SubpathPRD & aLightPrd, const Light & aLight, const float & aLightPickPdf,
                                                  const float & misVcWeightFactor)
{
    using namespace optix;

    float emissionPdfW;
    float directPdfW;
    float cosAtLight;
    aLightPrd.throughput = lightEmit(aLight, aLightPrd.randomState, aLightPrd.origin, aLightPrd.direction,
        emissionPdfW, directPdfW, cosAtLight);
    // vmarz?: do something similar as done for photon emission, emit towards scene when light far from scene?

    emissionPdfW *= aLightPickPdf;
    directPdfW *= aLightPickPdf;
    aLightPrd.throughput /= emissionPdfW;
    //lightPrd.isFinite = isDelta.isFinite ... vmarz?

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

    // e.g. if not delta ligth
    if (!aLight.isDelta)
    {
    	const float usedCosLight = aLight.isFinite ? cosAtLight : 1.f;
    	aLightPrd.dVC = vcmMis(usedCosLight / emissionPdfW);
        // dVC_1 = _g0 / ( p0_trace * p1 ) 
        // usedCosLight is part of _g0 - reverse pdf conversion factor!, uses outgoing cos not incident at next vertex,
        //    sqr(dist) from _g0 added after tracing
        // emissionPdfW = p0_trace * p1
    }

    // dVM_1 = dVC_1 / etaVCM
    aLightPrd.dVM = aLightPrd.dVC * misVcWeightFactor;
}


// Initialize camera payload - partial MIS terms [tech. rep. (31)-(33)]
optix::float3 __inline __device__ initCameraPayload(SubpathPRD & aCameraPrd, const Camera & aCamera, 
                                                    const optix::float2 & aPixelSizeFactor, const optix::uint & aVcmLightSubpathCount)
{
    using namespace optix;

    // pdf conversion factor from area on image plane to solid angle on ray
    float cosAtCamera = dot(normalize(aCamera.lookdir), aCameraPrd.direction);
    float imagePointToCameraDist = length(aCamera.lookdir) / cosAtCamera;
    float imageToSolidAngleFactor = sqr(imagePointToCameraDist) / cosAtCamera;

    float pixelArea = aPixelSizeFactor.x * aCamera.imagePlaneSize.x * aPixelSizeFactor.x * aCamera.imagePlaneSize.y;
    float areaSamplePdf = 1.f / pixelArea;

    // Needed if use different image point sampling techniques, see p0connect/p0trace in dVCM comment below
    //float p0connect = areaSamplePdf;      // cancel out
    //float p0trace = areaSamplePdf;        // cancel out
    float cameraPdf = areaSamplePdf * imageToSolidAngleFactor;

    // Initialize sub-path MIS quantities, partially [tech. rep. (31)-(33)]
    aCameraPrd.dVC = .0f;
    aCameraPrd.dVM = .0f;

    // dVCM = ( p0connect / p0trace ) * ( nLightSamples / p1 )
    // p0connect/p0trace - potentially different sampling techniques 
    //      p0connect - pdf for tecqhnique used when connecting to camera  during light tracing step
    //      p0trace - pdf for tecqhnique used when sampling a ray starting point
    // p1 = p1_ro * g1 = areaSamplePdf * imageToSolidAngleFactor * g1 [g1 added after tracing]
    // p0connect/p0trace cancel out in our case
    aCameraPrd.dVCM = vcmMis( aVcmLightSubpathCount / cameraPdf );

    //cameraPrd.specularPath = 1; // vmarz TODO ?
}



// Update MIS quantities before storing at the vertex, follows initialization on light [tech. rep. (31)-(33)]
// or scatter from surface [tech. rep. (34)-(36)]
optix::float3 __inline __device__ updateMisTermsOnHit(SubpathPRD & aLightPrd, const float & aCosThetaIn, const float & aRayLen)
{
    // sqr(dist) term from g in 1/p1 (or 1/pi), for dVC and dVM sqr(dist) terms of _g and pi cancel out
    aLightPrd.dVCM /= sqr(aRayLen);
    aLightPrd.dVCM *= vcmMis(aCosThetaIn);  // vmarz?: need abs here?
    aLightPrd.dVC *= vcmMis(aCosThetaIn);
    aLightPrd.dVM *= vcmMis(aCosThetaIn);

}


optix::float3 __inline __device__ updateMisTermsOnScatter(SubpathPRD & aLightPrd, const float & aCosThetaOut, const float & aBsdfDirPdfW,
                                                          const float & aBsdfRevPdfW, const float & aMisVcWeightFactor, const float & aMisVmWeightFactor)
{
    aLightPrd.dVC = vcmMis(aCosThetaOut / aBsdfDirPdfW) * ( // dVC = (g_i-1 / pi) * (etaVCM + dVCM_i-1 + _p_ro_i-2 * dVC_i-1)
        aLightPrd.dVC * vcmMis(aBsdfRevPdfW) +              // cosThetaOut part of g_i-1  [ _g reverse pdf conversion!, uses outgoing cosTheta]
        aLightPrd.dVCM + aMisVmWeightFactor);               //   !! sqr(dist) terms for _g_i-1 and gi of pi are the same and cancel out, hence NOT scaled after tracing]
                                                            // pi = bsdfDirPdfW * g1
    aLightPrd.dVM = vcmMis(aCosThetaOut / aBsdfDirPdfW) * ( // bsdfDirPdfW = _p_ro_i    [part of pi]
        aLightPrd.dVM * vcmMis(aBsdfRevPdfW) +              // bsdfRevPdfW = _p_ro_i-2
        aLightPrd.dVCM * aMisVcWeightFactor + 1.f);         // 
                                                            // dVM = (g_i-1 / pi) * (1 + dVCM_i-1/etaVCM + _p_ro_i-2 * dVM_i-1)
    aLightPrd.dVCM = vcmMis(1.f / aBsdfDirPdfW);            // cosThetaOut part of g_i-1 [_g reverse pdf conversion!, uses outgoing cosTheta]
                                                            //    !! sqr(dist) terms for _g_i-1 and gi of pi are the same and cancel out, hence NOT scaled after tracing]
                                                            //
                                                            // dVCM = 1 / pi
                                                            // pi = bsdfDirPdfW * g1 = _p_ro_i * g1 [only for dVCM sqe(dist) terms do not cancel out and are added after tracing]
}