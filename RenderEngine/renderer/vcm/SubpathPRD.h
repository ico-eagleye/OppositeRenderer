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
    RandomState randomState;
	float dVCM;
	float dVC;
	float dVM;
	//uint  mIsFiniteLight :  1; // Just generate by finite light
    //uint  mSpecularPath  :  1; // All scattering events so far were specular
};
