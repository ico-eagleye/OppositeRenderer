/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "config.h"
#include "renderer/Light.h"
#include "renderer/Camera.h"
#include "renderer/RayType.h"
#include "renderer/helpers/helpers.h"
#include "renderer/helpers/samplers.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/light.h"
#include "renderer/vcm/PathVertex.h"
#include "renderer/vcm/SubpathPRD.h"
#include "renderer/vcm/vcm.h"

using namespace optix;

rtDeclareVariable(rtObject, sceneRootObject, , );
rtDeclareVariable(Camera, camera, , );
rtBuffer<Light, 1> lights;
rtBuffer<float3, 2> outputBuffer;
rtBuffer<RandomState, 2> randomStates;
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim, );
//rtDeclareVariable(Sphere, sceneBoundingSphere, , );

// VCM
rtDeclareVariable(float2, pixelSizeFactor, , );
rtDeclareVariable(float, vcmMisVcWeightFactor, , ); // vmarz TODO set
rtDeclareVariable(float, vcmMisVmWeightFactor, , ); // vmarz TODO set
rtDeclareVariable(uint, vcmLightSubpathCount, , ); // vmarz TODO set


RT_PROGRAM void cameraPass()
{
    SubpathPRD cameraPrd;
    cameraPrd.randomState = randomStates[launchIndex];
    cameraPrd.throughput = make_float3(1.0f);
    cameraPrd.depth = 0;
    cameraPrd.done = 0;
    cameraPrd.dVC = 0;
    cameraPrd.dVM = 0;
    cameraPrd.dVCM = 0;   

    float2 screen = make_float2( outputBuffer.size() );
    float2 sample = getRandomUniformFloat2(&cameraPrd.randomState);             // jitter pixel pos
    float2 d = ( make_float2(launchIndex) + sample ) / screen * 2.0f - 1.0f;    // vmarz: map pixel pos to [-1,1]
    
    cameraPrd.origin = camera.eye;
    cameraPrd.direction = normalize(d.x*camera.camera_u + d.y*camera.camera_v + camera.lookdir);
    //modifyRayForDepthOfField(camera, rayOrigin, rayDirection, radiancePrd.randomState);     // vmarz TODO add ?
    Ray cameraRay = Ray(cameraPrd.origin, cameraPrd.direction, RayType::CAMERA_VCM, RAY_LEN_MIN, RT_DEFAULT_MAX );

    initCameraPayload(cameraPrd, camera, pixelSizeFactor, vcmLightSubpathCount);

    //// pdf conversion factor from area on image plane to solid angle on ray
    //float cosAtCamera = dot(normalize(camera.lookdir), cameraPrd.direction);
    //float imagePointToCameraDist = length(camera.lookdir) / cosAtCamera;
    //float imageToSolidAngleFactor = sqr(imagePointToCameraDist) / cosAtCamera;

    //// Needed if use different image point sampling techniques, see p0connect/p0trace in dVCM
    ////float pixelArea = pixelSizeFactor.x * camera.imagePlaneSize.x * pixelSizeFactor.x * camera.imagePlaneSize.y;
    ////float areaSamplePdf = 1.f / pixelArea;
    //float cameraPdfW =/* areaSamplePdf * */ imageToSolidAngleFactor; // areaPdf = 1/area = 1/1 = 1

    ////cameraPrd.specularPath = 1; // vmarz TODO ?
    //// Initialize sub-path MIS quantities, partially [tech. rep. (31)-(33)]
    //cameraPrd.dVC = .0f;
    //cameraPrd.dVM = .0f;
    ////float cameraConnectPdf = areaSamplePdf;
    //cameraPrd.dVCM = vcmMis(vcmLightSubpathCount / cameraPdfW); 
    //// dVCM = (p0connect/p0trace)*(nLightSamples/pi)
    //// p0connect/p0trace - potentially different sampling techniques 
    ////      p0connect - pdf for tecqhnique used when connecting to camera  during light tracing step
    ////      p0trace - pdf for tecqhnique used when sampling a ray starting point

    // Trace    
    for (int i=0;;i++)
    {
        //OPTIX_DEBUG_PRINT(cameraPrd.depth, "G %d - tra dir %f %f %f\n",
        //    i, cameraRay.direction.x, cameraRay.direction.y, cameraRay.direction.z);
        rtTrace( sceneRootObject, cameraRay, cameraPrd );
        
        // sample direct lightning

        // vertext connection

        // vertex merging

        if (cameraPrd.done)
        {
            //OPTIX_DEBUG_PRINT(cameraPrd.depth, "Stop trace \n");
            break;
        }

        // sample new dir

        cameraRay.origin = cameraPrd.origin;
        cameraRay.direction = cameraPrd.direction;

        //OPTIX_DEBUG_PRINT(cameraPrd.depth, "G %d - new org %f %f %f\n", i, cameraRay.origin.x, cameraRay.origin.y, cameraRay.origin.z);
        //OPTIX_DEBUG_PRINT(cameraPrd.depth, "G %d - new dir %f %f %f\n", i, cameraRay.direction.x, cameraRay.direction.y, cameraRay.direction.z);
    }

    randomStates[launchIndex] = cameraPrd.randomState;
}


rtDeclareVariable(SubpathPRD, cameraPrd, rtPayload, );
RT_PROGRAM void miss()
{
    cameraPrd.done = 1;
    OPTIX_DEBUG_PRINT(cameraPrd.depth, "Miss\n");
    //rtPrintf("%d %d: MISS depth %d ndir %f %f %f\n", launchIndex.x, launchIndex.y, cameraPrd.depth,
    //            cameraPrd.direction.x, cameraPrd.direction.y, cameraPrd.direction.z);
}


// Exception handler program
rtDeclareVariable(float3, exceptionErrorColor, , );
RT_PROGRAM void exception()
{
    rtPrintf("Exception VCM Camera ray! d: %d\n", cameraPrd.depth);
    rtPrintExceptionDetails();
    cameraPrd.throughput = make_float3(1,0,0);
}
