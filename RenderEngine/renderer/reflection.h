/*
    pbrt source code Copyright(c) 1998-2012 Matt Pharr and Greg Humphreys.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */
/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * BSDF, BxDF, Fresnel code is partially based on pbrt, idea of "fake virtual" functions via macros
 * borrowed from https://github.com/LittleCVR/MaoPPM
*/

#pragma once

#include <optix_world.h>
#include "renderer/device_common.h"
#include "renderer/helpers/helpers.h"

RT_FUNCTION float localCosTheta( const optix::float3 & w )
{
    return w.z;
}

RT_FUNCTION float localSinThetaSquared( const optix::float3 & w )
{
    return 1.0f - w.z*w.z;
}

RT_FUNCTION bool localIsSameHemisphere( const optix::float3 & wo, const optix::float3 & wi )
{
    return wo.z * wi.z > 0.0f;
}


RT_FUNCTION optix::float3 localReflect( const optix::float3 & w )
{
    return optix::make_float3(-w.x, -w.y, w.z);
}


class Fresnel 
{
public:
    enum Type {
        NoOp        = 1 << 0,
        Conductor   = 1 << 1,
        Dielectric  = 1 << 2
    };


public:
    RT_FUNCTION Fresnel(Type type) : m_type(type) {  }

    RT_FUNCTION Type type() const { return m_type; }

    // optix::float3 evaluate(float cosi) const;

private:
    Type  m_type;
};


class FresnelNoOp : public Fresnel 
{

public:
    RT_FUNCTION FresnelNoOp() : Fresnel(NoOp) {  }

public:
    RT_FUNCTION float evaluate(float) const
    {
        return 1.0f;
    }

};

//class FresnelConductor : public Fresnel 
//{
//    public:
//        RT_FUNCTION FresnelConductor(
//                const optix::float3 & eta, const optix::float3 & k) : Fresnel(Fresnel::Conductor),
//            m_eta(eta), m_k(k) {  }   
//
//        RT_FUNCTION optix::float3 evaluate(float cosi) const
//        {
//            cosi = fabsf(cosi);
//            const optix::float3 & eta = m_eta;
//            const optix::float3 & k   = m_k;
//            optix::float3 tmp = (eta*eta + k*k) * cosi*cosi;
//            optix::float3 Rparl2 = (tmp - (2.f * eta * cosi) + 1.f) /
//                (tmp + (2.f * eta * cosi) + 1.f); 
//            optix::float3 tmp_f = eta*eta + k*k;
//            optix::float3 Rperp2 =
//                (tmp_f - (2.f * eta * cosi) + cosi*cosi) /
//                (tmp_f + (2.f * eta * cosi) + cosi*cosi);
//            return (Rparl2 + Rperp2) / 2.f;
//        }
//
//    private:
//        optix::float3  m_eta;
//        optix::float3  m_k;
//};

class FresnelDielectric : public Fresnel 
{
public:
    RT_FUNCTION FresnelDielectric(float ei, float et) : Fresnel(Fresnel::Dielectric),
        eta_i(ei), eta_t(et) {  }

    RT_FUNCTION float evaluate(float cosi) const 
    {
        using namespace optix;

        // Compute Fresnel reflectance for dielectric
        cosi = optix::clamp(cosi, -1.0f, 1.0f);

        // Compute indices of refraction for dielectric
        bool entering = cosi > 0.0f;
        float ei = eta_i, et = eta_t;
        if (!entering) swap(ei, et);

        // Compute _sint_ using Snell's law
        float sint = ei/et * sqrtf(fmaxf(0.0f, 1.0f - cosi*cosi));
        if (sint >= 1.0f)
        {
            // Handle total internal reflection
            return 1.0f;
        }
        else 
        {
            cosi = fabsf(cosi);
            float cost = sqrtf(fmaxf(0.0f, 1.0f - sint*sint));
            float Rparl = 
                ((et * cosi) - (ei * cost)) /
                ((et * cosi) + (ei * cost));
            float Rperp = 
                ((ei * cosi) - (et * cost)) /
                ((ei * cosi) + (et * cost));
            return (Rparl*Rparl + Rperp*Rperp) * 0.5f;
        }
    }

public:  // should be private
    float eta_i, eta_t;
};

static const unsigned int  MAX_FRESNEL_SIZE  = sizeof(FresnelDielectric);