#pragma once
#include <optix_world.h>


__device__ __inline__ float localCosTheta(const optix::float3 & w)
{
    return w.z;
}

__device__ __inline__ float localSinThetaSquared(const optix::float3 & w)
{
    return 1.0f - w.z*w.z;
}
__device__ __inline__ bool localIsSameHemisphere(
        const optix::float3 & wo, const optix::float3 & wi)
{
    return wo.z * wi.z > 0.0f;
}
   