/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "renderer/RandomState.h"
struct PhotonPRD
{
    optix::float3 power;
    float weight;					// vmarz: initially 1, scaled by fmax(Kd) at every hit, used to stop tracing when small
    optix::uint pm_index;			// vmarz: index of the first photon stored by a given light path/thread
    optix::uint numStoredPhotons;
    optix::uint depth;
    RandomState randomState;
};
