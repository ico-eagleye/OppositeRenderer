/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include <optix.h>

// Applies MIS power
static __host__ __device__ __inline__ float vcmMis(const float & aPdf)
{
    // balance heuristic for now
    return aPdf;
}