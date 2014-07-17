/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#define OPTIX_PRINTF_DEF
#define OPTIX_PRINTFI_DEF
#define OPTIX_PRINTFID_DEF

#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include <cuda_runtime.h>
#include "renderer/helpers/helpers.h"
#include "config.h"
#include "renderer/Light.h"
#include "renderer/RayType.h"
#include "renderer/helpers/samplers.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/light.h"
#include "math/Sphere.h"
#include "renderer/vcm/LightVertex.h"
#include "renderer/vcm/SubpathPRD.h"
#include "renderer/vcm/vcm.h"
#include "renderer/vcm/mis.h"
#include "renderer/vcm/config_vcm.h"


#define OPTIX_PRINTF_ENABLED 0
#define OPTIX_PRINTFI_ENABLED 0
#define OPTIX_PRINTFID_ENABLED 0
#define OPTIX_PRINTFC_ENABLED 0
#define OPTIX_PRINTFCID_ENABLED 0


void initLightPayload(SubpathPRD & aLightPrd);

using namespace optix;

rtDeclareVariable(rtObject, sceneRootObject, , );
rtBuffer<RandomState, 2> randomStates;
rtBuffer<Light, 1> lights;
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim, );

rtDeclareVariable(uint, lightVertexCountEstimatePass, , );
rtBuffer<uint, 2> lightSubpathLengthBuffer;
//rtBuffer<uint, 3> lightSubpathVertexIndexBuffer;
//rtBuffer<LightVertex> lightVertexBuffer;

rtDeclareVariable(int, lightSubpathLengthBufferId, , ); // <uint, 2>
rtDeclareVariable(int, lightSubpathVertexIndexBufferId, , ); // <uint, 3>
rtDeclareVariable(int, lightVertexBufferId, , ); // <LightVertex>


RT_PROGRAM void lightPass()
{
    if (lightVertexCountEstimatePass)
        { OPTIX_PRINTFID(launchIndex, 0u, "\n\nGenCL - LIGHT ESTIMATE PASS -----------------------------------------------------------------\n"); }
    else
        { OPTIX_PRINTFID(launchIndex, 0u, "\n\nGenCL - LIGHT STORE PASS --------------------------------------------------------------------\n"); }

    SubpathPRD lightPrd;
    // Initialize payload and ray
    initLightPayload(lightPrd);
    Ray lightRay = Ray(lightPrd.origin, lightPrd.direction, RayType::LIGHT_VCM, RAY_LEN_MIN, RT_DEFAULT_MAX );

    for (int i=0;;i++)
    {
        //OPTIX_PRINTFI(lightPrd.depth, "G %d - tra dir %f %f %f\n",
        //    i, lightRay.direction.x, lightRay.direction.y, lightRay.direction.z);
        rtTrace( sceneRootObject, lightRay, lightPrd );

        if (lightPrd.done)
        {
            OPTIX_PRINTFID(launchIndex, lightPrd.depth, "GenCL - DONE LIGHT RAY\n\n");
            break;
        }

        lightRay.origin = lightPrd.origin;
        lightRay.direction = lightPrd.direction;

        //OPTIX_PRINTFI(lightPrd.depth, "G %d - new org %f %f %f\n", i, lightRay.origin.x, lightRay.origin.y, lightRay.origin.z);
        //OPTIX_PRINTFI(lightPrd.depth, "G %d - new dir %f %f %f\n", i, lightRay.direction.x, lightRay.direction.y, lightRay.direction.z);
    }

    randomStates[launchIndex] = lightPrd.randomState;
    lightSubpathLengthBuffer[launchIndex] = lightPrd.depth;
}




rtDeclareVariable(SubpathPRD, lightPrd, rtPayload, );
RT_PROGRAM void miss()
{
    lightPrd.done = true;
    //OPTIX_PRINTFI(lightPrd.depth, "Miss\n");
    OPTIX_PRINTFID(launchIndex, lightPrd.depth, "GenCL -       MISS dirW % 14f % 14f % 14f           from % 14f % 14f % 14f \n",
                      lightPrd.direction.x, lightPrd.direction.y, lightPrd.direction.z,
                      lightPrd.origin.x, lightPrd.origin.y, lightPrd.origin.z);
}


// Exception handler program
rtDeclareVariable(float3, exceptionErrorColor, , );
RT_PROGRAM void exception()
{
    rtPrintf("Exception Light ray! d: %d\n", lightPrd.depth);
    rtPrintExceptionDetails();
    lightPrd.throughput = make_float3(0,0,0);
}



rtDeclareVariable(float, misVcWeightFactor, , ); // 1/etaVCM
rtDeclareVariable(float, vertexPickPdf, , );


// Initialize light payload - throughput premultiplied with light radiance, partial MIS terms  [tech. rep. (31)-(33)]
RT_FUNCTION void initLightPayload(SubpathPRD & aLightPrd)
{
    using namespace optix;

    aLightPrd.launchIndex = launchIndex;
    aLightPrd.throughput = make_float3(1.f);
    aLightPrd.depth = 0.f;
    aLightPrd.dVC = 0.f;
    aLightPrd.dVM = 0.f;
    aLightPrd.dVCM = 0.f;
    aLightPrd.done = false;
    aLightPrd.isSpecularPath = true;
    aLightPrd.randomState = randomStates[launchIndex];
    lightSubpathLengthBuffer[launchIndex] = 0u;

    float *pVertPickPdf = NULL;
#if VCM_UNIFORM_VERTEX_SAMPLING
    aLightPrd.dVC_unif_vert = 0.f;
    pVertPickPdf = &vertexPickPdf;
#endif

    // vmarz TODO: pick based on light power
    int lightIndex = 0;
    if (1 < lights.size())
    {
        float sample = getRandomUniformFloat(&aLightPrd.randomState);
        lightIndex = intmin((int)(sample*lights.size()), int(lights.size()-1));
    }

    const Light light        = lights[lightIndex];
    const float lightPickPdf = 1.f / lights.size();

    float emissionPdfW;
    float directPdfW;
    float cosAtLight;
    aLightPrd.throughput = lightEmit(light, aLightPrd.randomState, aLightPrd.origin, aLightPrd.direction,
        emissionPdfW, directPdfW, cosAtLight, &aLightPrd.launchIndex);
    // vmarz?: do something similar as done for photon emission, emit towards scene when light far from scene?

    emissionPdfW *= lightPickPdf;
    directPdfW   *= lightPickPdf;
    aLightPrd.throughput /= emissionPdfW;
    aLightPrd.isGenByFiniteLight = light.isFinite;

    //lightPrd.isFinite = isDelta.isFinite ... vmarz?
    OPTIX_PRINTFID(aLightPrd.launchIndex, 0u, "GenLi - emission Pdf    % 14f     directPdfW % 14f     cosAtLight % 14f \n", 
        emissionPdfW, directPdfW, cosAtLight);
    OPTIX_PRINTFID(aLightPrd.launchIndex, 0u, "GenLi - prd throughput  % 14f % 14f % 14f\n", 
        aLightPrd.throughput.x, aLightPrd.throughput.y, aLightPrd.throughput.z);

    initLightMisTerms(aLightPrd, light, cosAtLight, directPdfW, emissionPdfW, misVcWeightFactor, pVertPickPdf);
}