/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "renderer/RandomState.h"
struct SubpathPRD
{
    optix::float3 origin;
    optix::float3 direction;
    optix::float3 throughput;
    optix::uint depth;
    uint seed;
    RandomState randomState;
    float dVCM;
    float dVC;
    float dVM;
    optix::uint done;
    optix::uint keepTracing; // vmarz: rtTrace() sometimes doesn't result in anyhit or miss program called
                             // hence can't use "done" condition as in path_trace sample 
                             // https://devtalk.nvidia.com/default/topic/754670/optix/rttrace-occasionally-results-in-nothing-no-call-to-any-hit-miss-or-exception-program-/
    //uint  mIsFiniteLight :  1; // Just generate by finite light
    //uint  mSpecularPath  :  1; // All scattering events so far were specular
};

// #define PRD_HIT_EMITTER (1<<31u)