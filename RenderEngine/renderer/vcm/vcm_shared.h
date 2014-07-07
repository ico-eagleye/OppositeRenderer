#pragma once
#include <optix.h>

// Applies MIS power
static __host__ __device__ __inline__ float vcmMis(const float & aPdf)
{
    // balance heuristic for now
    return aPdf;
}