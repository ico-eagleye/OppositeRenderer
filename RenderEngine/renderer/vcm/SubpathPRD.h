/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Valdis Vilcans
*/

#pragma once
#include "renderer/RandomState.h"
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "renderer/vcm/config_vcm.h"

struct SubpathPRD
{
    optix::float3 origin;
    optix::float3 direction;
    optix::float3 throughput;
    optix::float3 color;        // accumulated full path contributions
    optix::uint2 launchIndex;
    optix::uint  launchIndex1D;
    optix::uint depth;
    RandomState randomState;
    float dVCM;
    float dVC;
    float dVM;
#if VCM_UNIFORM_VERTEX_SAMPLING
    float dVC_unif_vert;       // dVC for uniform connection vertex sampling, account for vertex pick probability
#endif
    bool done;
    bool isGenByFiniteLight;    // Just generated by finite light
    bool isSpecularPath;        // Indicates all scattering events so far were specular
};
