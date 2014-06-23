/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "helpers.h"
#include <optix.h>
#include <optixu/optixu_math_namespace.h>

// Get a random direction from the hemisphere of direction around normalized normal, 
// sampled with the cosine distribution p(theta, phi) = cos(theta)/PI. In oPdfW returns PDF with 
// respect to solid angle measure
static __device__ __inline__ optix::float3 sampleUnitHemisphereCos(const optix::float3 & normal, const optix::float2& sample,
                                                                   float * oPdfW = NULL, float * oCosTheta = NULL)
{
    using namespace optix;

    float theta = acosf(sqrtf(sample.x));
    float phi = 2.0f * M_PIf *sample.y;
    float xs = sinf(theta) * cosf(phi);
    float ys = cosf(theta);
    float zs = sinf(theta) * sinf(phi);

    float3 U, V;
    createCoordinateSystem(normal, U, V);
    if (oPdfW)
        *oPdfW = ys * M_1_PIf;
    if (oCosTheta)
        *oCosTheta = ys;

    return optix::normalize(xs*U + ys*normal + zs*V);
}

// Sample unit hemisphere around (normalized) normal
static __device__ __inline__ optix::float3 sampleUnitHemisphere(const optix::float3 & normal, const optix::float2& sample)
{
    optix::float3 U, V;
    createCoordinateSystem( normal, U, V);
    float phi = 2.0f * M_PIf*sample.x;
    float r = sqrtf( sample.y );
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = 1.0f - x*x -y*y;
    z = z > 0.0f ? sqrtf(z) : 0.0f;
    return optix::normalize(U*x + V*y + normal*z);
}

static __device__ __inline__ optix::float3 sampleUnitSphere(const optix::float2& sample, float * oPdfW = NULL)
{
    optix::float3 v;
    v.z = sample.x;
    float t =  2*M_PIf*sample.y;
    float r = sqrtf(1.f-v.z*v.z);
    v.x = r*cosf(t);
    v.y = r*sinf(t);
    if (oPdfW)
        *oPdfW = 0.25f * M_1_PIf;

    return v;
}

static __device__ __inline__ optix::float2 sampleUnitDisc(const optix::float2& sample)
{
    float r = sqrtf(sample.x);
    float theta = 2.f*M_PIf*sample.y;
    float x = r*cosf(theta); // crashes with "defs/uses not defined for PTX instruction" without -use_fast_math flag on GTX770 CUDA v6 runtime
    float y = r*sinf(theta); // crashes with "defs/uses not defined for PTX instruction" without -use_fast_math flag on GTX770 CUDA v6 runtime
    return make_float2(x, y);
}

// Sample disc (normal must be normalized)
static __device__ float3 sampleDisc(const float2 & sample, const float3 & center, const float radius, const float3 & normal)
{
    float3 U, V;
    createCoordinateSystem( normal, U, V);
    float2 unitDisc = sampleUnitDisc(sample);
    return center + radius * ( U*unitDisc.x + V*unitDisc.y );
}

// Applies MIS power
static __device__ __inline__ float vcmMis(const float & aPdf)
{
    // balance heuristic for now
    return aPdf;
}

// <Sampling code from Optix SDK>
//Create ONB from normalaized vector
static __device__ __inline__ void createONB( 
    const optix::float3& n, optix::float3& U, optix::float3& V)
{
  using namespace optix;

  U = cross( n, make_float3( 0.0f, 1.0f, 0.0f ) );
  if ( dot(U, U) < 1.e-3f )
      U = cross( n, make_float3( 1.0f, 0.0f, 0.0f ) );
  U = normalize( U );
  V = cross( n, U );
}


optix::float3 __device__ __inline__ sampleHemisphereCosOptix(optix::float3 normal, optix::float2 rnd)
{
    using namespace optix;
    float3 p;
    cosine_sample_hemisphere(rnd.x, rnd.y, p);
    float3 v1, v2;
    createONB(normal, v1, v2);
    return v1 * p.x + v2 * p.y + normal * p.z;  
}
// </Sampling code from Optix SDK>
