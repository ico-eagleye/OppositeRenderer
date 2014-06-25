#pragma once
#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "renderer/helpers/samplers.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/light.h"
#include "renderer/Light.h"
#include "renderer/helpers/helpers.h"
#include "renderer/vcm/SubpathPRD.h"



optix::float3 __inline __device__ initLightSample(SubpathPRD & aLightPrd, const Light & aLight, const float & aLightPickPdf,
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