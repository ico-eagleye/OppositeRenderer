/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Valdis Vilcans
*/

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
#include "renderer/vcm/SubpathPRD.h"
#include "renderer/vcm/config_vcm.h"
#include "renderer/vcm/vcm_shared.h"

#define OPTIX_PRINTF_ENABLED 0
#define OPTIX_PRINTFI_ENABLED 0
#define OPTIX_PRINTFID_ENABLED 0

// Initialize light payload partial MIS terms  [tech. rep. (31)-(33)]
RT_FUNCTION void initLightMisTerms(SubpathPRD & aLightPrd, const Light & aLight, const float aCostAtLight,
                                    const float aDirectPdfW, const float aEmissionPdfW,
                                    const float misVcWeightFactor, const float const * aVertexPickPdf = NULL)
{
    using namespace optix;

    // Partial light sub-path MIS quantities. [tech. rep. (31)-(33)]
    // Evaluation is completed after tracing the emission ray
    aLightPrd.dVCM = vcmMis(aDirectPdfW / aEmissionPdfW);
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
        const float usedCosLight = aLight.isFinite ? aCostAtLight : 1.f;
        aLightPrd.dVC = vcmMis(usedCosLight / aEmissionPdfW);
        // dVC_1 = _g0 / ( p0_trace * p1 ) 
        // usedCosLight is part of _g0 - reverse pdf conversion factor!, uses outgoing cos not incident at next vertex,
        //    sqr(dist) from _g0 added after tracing
        // emissionPdfW = p0_trace * p1

#if VCM_UNIFORM_VERTEX_SAMPLING
        aLightPrd.dVC_unif_vert = aLightPrd.dVC;
#endif
    }
    else
    {
        aLightPrd.dVC = 0.f;
    }

    // dVM_1 = dVC_1 / etaVCM
    aLightPrd.dVM = aLightPrd.dVC * misVcWeightFactor;

#if VCM_UNIFORM_VERTEX_SAMPLING
    if (aVertexPickPdf)
        aLightPrd.dVM *= *aVertexPickPdf; // should divide etaVCM, bust since it is denominator, we just multiply
    // aVertexPickPdf pointer passed only when using uniform vertex sampling from buffer
#endif
}


// Initialize camera payload partial MIS terms [tech. rep. (31)-(33)]
RT_FUNCTION void initCameraMisTerms(SubpathPRD & aCameraPrd, const float aCameraPdfW, const optix::uint aVcmLightSubpathCount)
{
    // Initialize sub-path MIS quantities, partially [tech. rep. (31)-(33)]
    aCameraPrd.dVC = .0f;
    aCameraPrd.dVM = .0f;
#if VCM_UNIFORM_VERTEX_SAMPLING
    aCameraPrd.dVC_unif_vert = aCameraPrd.dVC;
#endif

    // dVCM = ( p0connect / p0trace ) * ( nLightSamples / p1 )
    // p0connect/p0trace - potentially different sampling techniques 
    //      p0connect - pdf for technique used when connecting to camera during light tracing step
    //      p0trace - pdf for technique used when sampling a ray starting point
    // p1 = aCameraPdfW = p1_ro * g1 = areaSamplePdf * imageToSolidAngleFactor * g1 [g1 added after tracing]
    // p0connect/p0trace cancel out in our case
    aCameraPrd.dVCM = vcmMis( aVcmLightSubpathCount / aCameraPdfW );
    //OPTIX_PRINTFID(aCameraPrd.launchIndex, "Gen C - init  - dVCM %f lightSubCount %d camPdf %f\n", aCameraPrd.dVCM, 
    //    aVcmLightSubpathCount, cameraPdf);

    aCameraPrd.isSpecularPath = true;
}



// Update MIS quantities before storing at the vertex, follows initialization on light [tech. rep. (31)-(33)]
// or scatter from surface [tech. rep. (34)-(36)]
RT_FUNCTION void updateMisTermsOnHit(SubpathPRD & aLightPrd, const float aCosThetaIn, const float aRayLen)
{
    // infinite lights potentially need additional handling here if MIS handled via solid angle integration [tech. rep. Section 5.1]

    // sqr(dist) term from g in 1/p1 (or 1/pi), for dVC and dVM sqr(dist) terms of _g and pi cancel out
    aLightPrd.dVCM *= vcmMis(sqr(aRayLen));
    aLightPrd.dVCM /= vcmMis(aCosThetaIn);
    aLightPrd.dVC  /= vcmMis(aCosThetaIn);
    aLightPrd.dVM  /= vcmMis(aCosThetaIn);
#if VCM_UNIFORM_VERTEX_SAMPLING
    aLightPrd.dVC_unif_vert  /= vcmMis(aCosThetaIn);
#endif
}



#define OPTIX_PRINTF_ENABLED 0
#define OPTIX_PRINTFI_ENABLED 0
#define OPTIX_PRINTFID_ENABLED 0
// nvcc of Cuda 6 gives error below if updateMisTermsOnScatter() is inlined after call to lightVertex.bsdf.sampleF or vcmSampleF
//“PHINode should have one entry for each predecessor of its parent basic block! 
//%__cuda_local_var_528573_11_non_const_bsdfDirPdfW.4 = phi float [ %__cuda_local_var_528573_11_non_const_bsdfDirPdfW.1609, %568 ], [ %580, %578 ], [ %__cuda_local_var_528573_11_non_const_bsdfDirPdfW.1609, %568 ], !dbg !515”

// Initializes MIS terms for next event, partial implementation of [tech. rep. (34)-(36)], completed on hit
RT_FUNCTION void updateMisTermsOnScatter(SubpathPRD & aPathPrd, const float & aCosThetaOut, const float & aBsdfDirPdfW,
                                         const float & aBsdfRevPdfW, const float & aMisVcWeightFactor, const float & aMisVmWeightFactor,
                                         BxDF::Type aSampledEvent, const float const * aVertexPickPdf = NULL)
{
    float vertPickPdf = aVertexPickPdf ?  (*aVertexPickPdf) : 1.f;
    const float dVC = aPathPrd.dVC;
    const float dVM = aPathPrd.dVM;
    const float dVCM = aPathPrd.dVCM;

    if (aSampledEvent & BxDF::Specular)
    {
        aPathPrd.dVCM = 0.f;
        // pdfs must be the same for specular event and so cancel out
        aPathPrd.dVC *= vcmMis(aCosThetaOut); // * (aBsdfRevPdfW / aBsdfDirPdfW)
        aPathPrd.dVM *= vcmMis(aCosThetaOut); // * (aBsdfRevPdfW / aBsdfDirPdfW)
        return;
    }

    // dVC = (g_i-1 / pi) * (etaVCM + dVCM_i-1 + _p_ro_i-2 * dVC_i-1)
    // cosThetaOut part of g_i-1  [ _g reverse pdf conversion!, uses outgoing cosTheta]
    //   !! sqr(dist) terms for _g_i-1 and gi of pi are the same and cancel out, hence NOT scaled after tracing]
    // pi = bsdfDirPdfW * g1
    // bsdfDirPdfW = _p_ro_i    [part of pi]
    // bsdfRevPdfW = _p_ro_i-2
    aPathPrd.dVC = vcmMis(aCosThetaOut / aBsdfDirPdfW) * ( 
        aPathPrd.dVC * vcmMis(aBsdfRevPdfW) +              
        aPathPrd.dVCM + aMisVmWeightFactor);               

#if VCM_UNIFORM_VERTEX_SAMPLING
    const float dVC_unif_vert = aPathPrd.dVC_unif_vert;
    aPathPrd.dVC_unif_vert = vcmMis(aCosThetaOut / aBsdfDirPdfW) * ( 
        aPathPrd.dVC_unif_vert * vcmMis(aBsdfRevPdfW) +              
        aPathPrd.dVCM + aMisVmWeightFactor / vertPickPdf);
#endif

    // dVM = (g_i-1 / pi) * (1 + dVCM_i-1/etaVCM + _p_ro_i-2 * dVM_i-1)
    // cosThetaOut part of g_i-1 [_g reverse pdf conversion!, uses outgoing cosTheta]
    //    !! sqr(dist) terms for _g_i-1 and gi of pi are the same and cancel out, hence NOT scaled after tracing]
    aPathPrd.dVM = vcmMis(aCosThetaOut / aBsdfDirPdfW) * ( 
        aPathPrd.dVM * vcmMis(aBsdfRevPdfW) +              
        aPathPrd.dVCM * aMisVcWeightFactor * vertPickPdf + 1.f ); // vertPickPdf should divide etaVCM which is inverse in aMisVcWeightFactor
    
    // dVCM = 1 / pi
    // pi = bsdfDirPdfW * g1 = _p_ro_i * g1 [only for dVCM sqe(dist) terms do not cancel out and are added after tracing]
    aPathPrd.dVCM = vcmMis(1.f / aBsdfDirPdfW);
}


#undef OPTIX_PRINTF_ENABLED
#undef OPTIX_PRINTFI_ENABLED
#undef OPTIX_PRINTFID_ENABLED