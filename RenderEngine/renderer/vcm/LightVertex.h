/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Valdis Vilcans
*/

#pragma once
//#include <optix_world.h>
#include "renderer/BSDF.h"
#include "optix.h"

struct LightVertex
{
    optix::float3   hitPoint;
    optix::float3   throughput;
    VcmBSDF         bsdf;
    optix::uint2    launchIndex;    // for debug TODO remove
    optix::uint     pathLen;
    float           dVCM;
    float           dVC;
    float           dVM;
//#if VCM_UNIFORM_VERTEX_SAMPLING
//    float   dVC_unif_vert;
//#endif
};