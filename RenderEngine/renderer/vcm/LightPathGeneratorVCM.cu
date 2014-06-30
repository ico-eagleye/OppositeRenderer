/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include <cuda_runtime.h>
#include "config.h"
#include "renderer/Light.h"
#include "renderer/RayType.h"
#include "renderer/helpers/helpers.h"
#include "renderer/helpers/samplers.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/light.h"
#include "math/Sphere.h"

#include "renderer/vcm/LightVertex.h"
#include "renderer/vcm/SubpathPRD.h"
#include "renderer/vcm/vcm.h"


using namespace optix;

rtDeclareVariable(rtObject, sceneRootObject, , );
rtBuffer<RandomState, 2> randomStates;
rtBuffer<Light, 1> lights;
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim, );
rtDeclareVariable(Sphere, sceneBoundingSphere, , );

#if ENABLE_RENDER_DEBUG_OUTPUT
rtBuffer<unsigned int, 2> debugPhotonPathLengthBuffer;
#endif

rtDeclareVariable(uint, lightVertexCountEstimatePass, , );
rtDeclareVariable(float, misVcWeightFactor, , ); // 1/etaVCM
//rtDeclareVariable(float, misVmWeightFactor, , ); // etaVCM

rtBuffer<uint, 2> lightVertexCountBuffer;

RT_PROGRAM void lightPass()
{
    SubpathPRD lightPrd;
    lightPrd.depth = 0;
    lightPrd.done = 0;
    lightPrd.dVC = 0;
    lightPrd.dVM = 0;
    lightPrd.dVCM = 0;
    lightPrd.randomState = randomStates[launchIndex];
    if (lightVertexCountEstimatePass)
        lightVertexCountBuffer[launchIndex] = 0u;

    // vmarz TODO: pick based on light power
    int lightIndex = 0;
    if (1 < lights.size())
    {
        float sample = getRandomUniformFloat(&lightPrd.randomState);
        lightIndex = intmin((int)(sample*lights.size()), int(lights.size()-1));
    }

    const Light light = lights[lightIndex];
    const float inverseLightPickPdf = lights.size();
    const float lightPickPdf = 1.f / lights.size();

    // Initialize payload and ray
    initLightPayload(lightPrd, light, lightPickPdf, misVcWeightFactor);
    Ray lightRay = Ray(lightPrd.origin, lightPrd.direction, RayType::LIGHT_VCM, RAY_LEN_MIN, RT_DEFAULT_MAX );

    for (int i=0;;i++)
    {
        //OPTIX_PRINTFI(lightPrd.depth, "G %d - tra dir %f %f %f\n",
        //    i, lightRay.direction.x, lightRay.direction.y, lightRay.direction.z);
        rtTrace( sceneRootObject, lightRay, lightPrd );

        if (lightPrd.done)
        {
            //OPTIX_PRINTFI(lightPrd.depth, "Stop trace \n");
            break;
        }

        lightRay.origin = lightPrd.origin;
        lightRay.direction = lightPrd.direction;

        //OPTIX_PRINTFI(lightPrd.depth, "G %d - new org %f %f %f\n", i, lightRay.origin.x, lightRay.origin.y, lightRay.origin.z);
        //OPTIX_PRINTFI(lightPrd.depth, "G %d - new dir %f %f %f\n", i, lightRay.direction.x, lightRay.direction.y, lightRay.direction.z);
    }

    randomStates[launchIndex] = lightPrd.randomState;
}



rtDeclareVariable(SubpathPRD, lightPrd, rtPayload, );
RT_PROGRAM void miss()
{
    lightPrd.done = 1;
    //OPTIX_PRINTFI(lightPrd.depth, "Miss\n");
    //OPTIX_PRINTFI(lightPrd.depth, "%d %d: MISS depth %d ndir %f %f %f\n", launchIndex.x, launchIndex.y, lightPrd.depth,
    //            lightPrd.direction.x, lightPrd.direction.y, lightPrd.direction.z);
}


// Exception handler program
rtDeclareVariable(float3, exceptionErrorColor, , );
RT_PROGRAM void exception()
{
    rtPrintf("Exception Light ray! d: %d\n", lightPrd.depth);
    rtPrintExceptionDetails();
    lightPrd.throughput = make_float3(0,0,0);
}