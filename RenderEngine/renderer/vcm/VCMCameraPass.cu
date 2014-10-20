/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Valdis Vilcans
*/

#define OPTIX_PRINTF_DEF
#define OPTIX_PRINTFI_DEF
#define OPTIX_PRINTFID_DEF
#define OPTIX_PRINTFCID_DEF

#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "device_functions.h"
#include "renderer/helpers/helpers.h"
#include "config.h"
#include "renderer/Light.h"
#include "renderer/Camera.h"
#include "renderer/RayType.h"
#include "renderer/helpers/samplers.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/light.h"
#include "renderer/vcm/LightVertex.h"
#include "renderer/vcm/SubpathPRD.h"
#include "renderer/vcm/vcm.h"
#include "renderer/vcm/mis.h"

#define OPTIX_PRINTF_ENABLED 1
#define OPTIX_PRINTFI_ENABLED 1
#define OPTIX_PRINTFID_ENABLED 1

void initCameraPayload(SubpathPRD & aCameraPrd);

using namespace optix;

rtDeclareVariable(rtObject, sceneRootObject, , );
rtBuffer<Light, 1> lights;
rtBuffer<float3, 2> outputBuffer;                   // TODO change to float4
rtDeclareVariable(uint, localIterationNumber, , );
rtBuffer<RandomState, 2> randomStates;
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim, );

#define OPTIX_PRINTF_ENABLED 0
#define OPTIX_PRINTFI_ENABLED 0
#define OPTIX_PRINTFID_ENABLED 0
#define OPTIX_PRINTFCID_ENABLED 0

RT_PROGRAM void cameraPass()
{
    OPTIX_PRINTFID(launchIndex, 0u, "\n\nGen C - CAMERA PASS -------------------------------------------------------------------------\n");
    SubpathPRD cameraPrd;
    initCameraPayload(cameraPrd);
    Ray cameraRay = Ray(cameraPrd.origin, cameraPrd.direction, RayType::CAMERA_VCM, RAY_LEN_MIN, RT_DEFAULT_MAX );

    OPTIX_PRINTFID(launchIndex, 0u, "Gen C - start - dVCM %f\n", cameraPrd.dVCM);

    // Trace    
    for (int i=0;;i++)
    {
        rtTrace( sceneRootObject, cameraRay, cameraPrd );

        if (cameraPrd.done)
        {
            break;
        }

        cameraRay.origin = cameraPrd.origin;
        cameraRay.direction = cameraPrd.direction;
    }

    float3 bufColor = outputBuffer[launchIndex];
    bufColor = bufColor + cameraPrd.color;
    float3 avgColor = bufColor / (localIterationNumber + 1);
    outputBuffer[launchIndex] = bufColor;
    randomStates[launchIndex] = cameraPrd.randomState;
}





rtDeclareVariable(SubpathPRD, cameraPrd, rtPayload, );
RT_PROGRAM void miss()
{
    cameraPrd.done = true;
}


// Exception handler program
rtDeclareVariable(float3, exceptionErrorColor, , );
RT_PROGRAM void exception()
{
    rtPrintf("Exception VCM Camera ray! d: %d\n", cameraPrd.depth);
    rtPrintExceptionDetails();
    //cameraPrd.throughput = make_float3(0,0,0);
}


rtDeclareVariable(Camera,   camera, , );
rtDeclareVariable(float2,   pixelSizeFactor, , );
rtDeclareVariable(uint,     lightSubpathCount, , );

// Initialize camera payload - partial MIS terms [tech. rep. (31)-(33)]
RT_FUNCTION void initCameraPayload(SubpathPRD & aCameraPrd)
{
    aCameraPrd.launchIndex   = launchIndex;
    aCameraPrd.launchIndex1D = getBufIndex1D(launchIndex, launchDim);
    aCameraPrd.randomState = randomStates[launchIndex];
    aCameraPrd.throughput = make_float3(1.0f);
    aCameraPrd.color = make_float3(0.0f);
    aCameraPrd.depth = 0;
    aCameraPrd.dVC = 0;
    aCameraPrd.dVM = 0;
    aCameraPrd.dVCM = 0;
    aCameraPrd.isSpecularPath = true;
#if VCM_UNIFORM_VERTEX_SAMPLING
    aCameraPrd.dVC_unif_vert = 0;
#endif
    aCameraPrd.done = false;

    float2 screen = make_float2( outputBuffer.size() );
    float2 sample = getRandomUniformFloat2(&aCameraPrd.randomState);            // jitter pixel pos
    float2 d = ( make_float2(launchIndex) + sample ) / screen * 2.0f - 1.0f;    // vmarz: map pixel pos to [-1,1]
    
    aCameraPrd.origin = camera.eye;
    aCameraPrd.direction = normalize(d.x*camera.camera_u + d.y*camera.camera_v + camera.lookdir);

    // pdf conversion factor from area on image plane to solid angle on ray
    float cosAtCamera = dot(normalize(camera.lookdir), aCameraPrd.direction);
    float distToImagePlane = length(camera.lookdir);
    float imagePointToCameraDist = distToImagePlane / cosAtCamera;
    float imageToSolidAngleFactor = sqr(imagePointToCameraDist) / cosAtCamera;

    float pixelArea = pixelSizeFactor.x * camera.imagePlaneSize.x * pixelSizeFactor.x * camera.imagePlaneSize.y;
    float areaSamplePdf = 1.f / pixelArea;

    // Needed if use different image point sampling techniques, see p0connect/p0trace in dVCM comment below
    //float p0connect = areaSamplePdf;      // cancel out
    //float p0trace = areaSamplePdf;        // cancel out
    float cameraPdfW = areaSamplePdf * imageToSolidAngleFactor;
    initCameraMisTerms(aCameraPrd, cameraPdfW, lightSubpathCount /*, p0trace, p0connect */);
}