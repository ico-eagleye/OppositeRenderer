// Partially borrowed from https://github.com/LittleCVR/MaoPPM

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
    RT_FUNCTION optix::float3 evaluate(float) const
    {
        return optix::make_float3(1.0f);
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

    RT_FUNCTION optix::float3 evaluate(float cosi) const 
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
        if (sint >= 1.0f) {
            // Handle total internal reflection
            return optix::make_float3(1.0f);
        } else {
            cosi = fabsf(cosi);
            float cost = sqrtf(fmaxf(0.0f, 1.0f - sint*sint));
            optix::float3 Rparl = optix::make_float3(
                ((et * cosi) - (ei * cost)) /
                ((et * cosi) + (ei * cost)));
            optix::float3 Rperp = optix::make_float3(
                ((ei * cosi) - (et * cost)) /
                ((ei * cosi) + (et * cost)));
            return (Rparl*Rparl + Rperp*Rperp) / 2.0f;
        }
    }

public:  // should be private
    float eta_i, eta_t;
};

static const unsigned int  MAX_FRESNEL_SIZE  = sizeof(FresnelDielectric);