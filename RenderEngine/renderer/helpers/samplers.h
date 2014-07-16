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
RT_FUNCTION static optix::float3 sampleUnitHemisphereCos(const optix::float3 & normal, const optix::float2& sample,
                                                         float * oPdfW = NULL, float * oCosTheta = NULL, bool biasSmallCosine = 0)
{
    using namespace optix;

    float theta = acosf(sqrtf(sample.x));
    float phi = 2.0f * M_PIf *sample.y;
    float xs = sinf(theta) * cosf(phi);
    float ys = cosf(theta);
    float zs = sinf(theta) * sinf(phi);

    float3 U, V;
    createCoordinateSystem(normal, U, V);
    if (biasSmallCosine && ys < EPS_COSINE) // otherwise sometimes it is 0
        ys = EPS_COSINE;
    if (oPdfW)
        *oPdfW = ys * M_1_PIf;
    if (oCosTheta)
        *oCosTheta = ys;

    return optix::normalize(xs*U + ys*normal + zs*V);
}

// Sample unit hemisphere around (normalized) normal
RT_FUNCTION static optix::float3 sampleUnitHemisphere(const optix::float3 & normal, const optix::float2& sample)
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

RT_FUNCTION static optix::float3 sampleUnitSphere(const optix::float2& sample, float * oPdfW = NULL)
{
    optix::float3 v;
    v.z = 1.f - 2.f * sample.x; // vmarz: needs to be in range [-1,1], actually is (-1,1) due random number range
    float phi =  2*M_PIf*sample.y;
    float r = sqrtf(1.f - v.z * v.z);
    v.x = r * cosf(phi);
    v.y = r * sinf(phi);

    if (oPdfW)
        *oPdfW = 0.25f * M_1_PIf;

    return v;
}

RT_FUNCTION static optix::float2 sampleUnitDisc(const optix::float2& sample)
{
    float r = sqrtf(sample.x);
    float theta = 2.f*M_PIf*sample.y;
    float x = r*cosf(theta); // crashes with "defs/uses not defined for PTX instruction" without -use_fast_math flag on GTX770 CUDA v6 runtime
    float y = r*sinf(theta); // crashes with "defs/uses not defined for PTX instruction" without -use_fast_math flag on GTX770 CUDA v6 runtime
    return make_float2(x, y);
}

// Sample disc (normal must be normalized)
RT_FUNCTION static float3 sampleDisc(const float2 & sample, const float3 & center, const float radius, const float3 & normal)
{
    float3 U, V;
    createCoordinateSystem( normal, U, V);
    float2 unitDisc = sampleUnitDisc(sample);
    return center + radius * ( U*unitDisc.x + V*unitDisc.y );
}


RT_FUNCTION float CosHemispherePdfW(const optix::float3  &aNormal, const optix::float3  &aDirection)
{
    return maxf(0.f, optix::dot(aNormal, aDirection)) * M_1_PIf;
}


RT_FUNCTION float powerCosHemispherePdfW( const optix::float3 & aNormal,
                                          const optix::float3 & aDirection,
                                          const float           aPower )
{
    const float cosTheta = optix::fmaxf(0.f, optix::dot(aNormal, aDirection));
    return (aPower + 1.f) * powf(cosTheta, aPower) * (M_1_PIf * 0.5f);
}

RT_FUNCTION optix::float3 samplePowerCosHemisphereW( const optix::float2 & aSamples,
                                                     const float           aPower,
                                                     float               * oPdfW )
{
    using namespace optix;
    const float term1 = 2.f * M_1_PIf * aSamples.x;
    const float term2 = powf(aSamples.y, 1.f / (aPower + 1.f));
    const float term3 = sqrt(1.f - term2 * term2);

    if (oPdfW)
    {
        *oPdfW = (aPower + 1.f) * powf(term2, aPower) * (0.5f * M_1_PIf);
    }

    return make_float3( cosf(term1) * term3, sinf(term1) * term3, term2);
}


//////////////////////////////////////////////////////////////////////////
// Utilities for converting PDF between Area (A) and Solid angle (W)
// WtoA = PdfW * cosine / distance_squared
// AtoW = PdfA * distance_squared / cosine
RT_FUNCTION float pdfWtoA( const float aPdfW, const float aDist, const float aCosThere )
{
    return aPdfW * std::abs(aCosThere) / sqr(aDist);
}

RT_FUNCTION float pdfAtoW( const float aPdfA, const float aDist, const float aCosThere )
{
    return aPdfA * sqr(aDist) / std::abs(aCosThere);
}
