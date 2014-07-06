/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#define OPTIX_PRINTFID_DISABLE
#define OPTIX_PRINTFI_DISABLE
#define OPTIX_PRINTFIALL_DISABLE

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
#include "renderer/vcm/LightVertex.h"
#include "renderer/vcm/SubpathPRD.h"
#include "renderer/vcm/vcm.h"

using namespace optix;

rtDeclareVariable(rtObject, sceneRootObject, , );
rtDeclareVariable(Camera, camera, , );
rtBuffer<Light, 1> lights;
rtBuffer<float3, 2> outputBuffer;                   // TODO change to float4
rtDeclareVariable(uint, localIterationNumber, , );
rtBuffer<RandomState, 2> randomStates;
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim, );
//rtDeclareVariable(Sphere, sceneBoundingSphere, , );

// VCM
rtDeclareVariable(float2, pixelSizeFactor, , );
rtDeclareVariable(float, vcmMisVcWeightFactor, , ); // vmarz TODO set
rtDeclareVariable(float, vcmMisVmWeightFactor, , ); // vmarz TODO set
rtDeclareVariable(uint, vcmLightSubpathCount, , ); // vmarz TODO set

static __device__ __inline__ float3 averageInNewRadiance(const float3 newRadiance, const float3 oldRadiance, const float localIterationNumber)
{
    if(localIterationNumber >= 1)
    {
        return oldRadiance + (newRadiance-oldRadiance)/(localIterationNumber+1);
    }
    else
    {
        return newRadiance;
    }
}


RT_PROGRAM void cameraPass()
{
    //if (launchIndex.x != 0 || launchIndex.y != 0) return;
    SubpathPRD cameraPrd;
    cameraPrd.randomState = randomStates[launchIndex];
    cameraPrd.throughput = make_float3(1.0f);
    cameraPrd.color = make_float3(0.0f);
    cameraPrd.depth = 0;
    cameraPrd.done = 0;
    cameraPrd.dVC = 0;
    cameraPrd.dVM = 0;
    cameraPrd.dVCM = 0;
    cameraPrd.launchIndex = launchIndex;

    float2 screen = make_float2( outputBuffer.size() );
    float2 sample = getRandomUniformFloat2(&cameraPrd.randomState);             // jitter pixel pos
    float2 d = ( make_float2(launchIndex) + sample ) / screen * 2.0f - 1.0f;    // vmarz: map pixel pos to [-1,1]
    
    cameraPrd.origin = camera.eye;
    cameraPrd.direction = normalize(d.x*camera.camera_u + d.y*camera.camera_v + camera.lookdir);
    //modifyRayForDepthOfField(camera, rayOrigin, rayDirection, radiancePrd.randomState);     // vmarz TODO add ?
    Ray cameraRay = Ray(cameraPrd.origin, cameraPrd.direction, RayType::CAMERA_VCM, RAY_LEN_MIN, RT_DEFAULT_MAX );

    initCameraPayload(cameraPrd, camera, pixelSizeFactor, vcmLightSubpathCount);
    OPTIX_PRINTFI(0, "Gen C - start - dVCM %f\n", cameraPrd.dVCM);

    // Trace    
    for (int i=0;;i++)
    {
        //OPTIX_PRINTFI(cameraPrd.depth, "G %d - tra dir %f %f %f\n",
        //    i, cameraRay.direction.x, cameraRay.direction.y, cameraRay.direction.z);
        rtTrace( sceneRootObject, cameraRay, cameraPrd );
        
        // sample direct lightning

        // vertext connection

        // vertex merging

        if (cameraPrd.done)
        {
            //OPTIX_PRINTFI(cameraPrd.depth, "Stop trace \n");
            break;
        }

        // sample new dir

        cameraRay.origin = cameraPrd.origin;
        cameraRay.direction = cameraPrd.direction;

        //OPTIX_PRINTFI(cameraPrd.depth, "G %d - new org %f %f %f\n", i, cameraRay.origin.x, cameraRay.origin.y, cameraRay.origin.z);
        //OPTIX_PRINTFI(cameraPrd.depth, "G %d - new dir %f %f %f\n", i, cameraRay.direction.x, cameraRay.direction.y, cameraRay.direction.z);
    }

    float3 avgColor = averageInNewRadiance(cameraPrd.color, outputBuffer[launchIndex], localIterationNumber);
    OPTIX_PRINTFI(cameraPrd.depth, "Gen C - DONE colr % 14f % 14f % 14f\n", cameraPrd.color.x, cameraPrd.color.y, cameraPrd.color.z);
    OPTIX_PRINTFI(cameraPrd.depth, "             avg  % 14f % 14f % 14f\n", avgColor.x, avgColor.y, avgColor.z);

    //OPTIX_PRINTF("%d , %d - d %d - iter %d prd.color %f %f %f avColor %f %f %f\n", 
    //    launchIndex.x, launchIndex.y, cameraPrd.depth, localIterationNumber,
    //    cameraPrd.color.x, cameraPrd.color.y, cameraPrd.color.z, avgColor.x, avgColor.y, avgColor.z);

    outputBuffer[launchIndex] = avgColor;
    randomStates[launchIndex] = cameraPrd.randomState;
}


rtDeclareVariable(SubpathPRD, cameraPrd, rtPayload, );
RT_PROGRAM void miss()
{
    cameraPrd.done = 1;
    //OPTIX_PRINTFI(cameraPrd.depth, "Miss\n");
    OPTIX_PRINTFI(cameraPrd.depth, "Gen C -      MISS dirW % 14f % 14f % 14f           from % 14f % 14f % 14f \n",
                      cameraPrd.direction.x, cameraPrd.direction.y, cameraPrd.direction.z,
                      cameraPrd.origin.x, cameraPrd.origin.y, cameraPrd.origin.z);
}


// Exception handler program
rtDeclareVariable(float3, exceptionErrorColor, , );
RT_PROGRAM void exception()
{
    rtPrintf("Exception VCM Camera ray! d: %d\n", cameraPrd.depth);
    rtPrintExceptionDetails();
    cameraPrd.throughput = make_float3(1,0,0);
}
