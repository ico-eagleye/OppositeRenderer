// Partially borrowed from https://github.com/LittleCVR/MaoPPM

#pragma once

#include <optixu/optixu_math_namespace.h>
#include "renderer/device_common.h"
#include "renderer/reflection.h"
#include "renderer/helpers/samplers.h"
#include "config.h"


// Cuda doesn't support virtual functions, hence resort to macro below
#define CALL_FRESNEL_CONST_VIRTUAL_FUNCTION(lvalue, op, fresnel, function, ...) \
    if (fresnel->type() & Fresnel::NoOp) \
    lvalue op reinterpret_cast<const FresnelNoOp *>(fresnel)->function(__VA_ARGS__); \
    else if (fresnel->type() & Fresnel::Dielectric) \
    lvalue op reinterpret_cast<const FresnelDielectric *>(fresnel)->function(__VA_ARGS__);


class BxDF 
{
public:
    enum Type
    {
        Null            = 0,
        // basic types,
        Reflection      = 1 << 0,
        Transmission    = 1 << 1,
        Diffuse         = 1 << 2,
        Glossy          = 1 << 3,
        Specular        = 1 << 4,
        AllType         = Diffuse | Glossy | Specular,
        AllReflection   = Reflection | AllType,
        AllTransmission = Transmission | AllType,
        All             = AllReflection | AllTransmission,
        // BxDF types
        Lambertian            = 1 << 5,
        SpecularReflection    = 1 << 6,
        SpecularTransmission  = 1 << 7
    };

private:
    Type  _type;

public:
    RT_FUNCTION BxDF(Type type) : _type(type) {  }

    RT_FUNCTION Type type() const { return _type; }

    // Because we compress the basic types and BxDF types in a single _type variable, it is necessary to AND All first.
    RT_FUNCTION bool matchFlags(Type type) const
    {
        return (_type & All & type) == (_type & All);
    }

    static RT_FUNCTION bool matchFlags(Type flagsToCheck, Type flagsToCheckFor)
    {
        return (flagsToCheck & All & flagsToCheckFor) == (flagsToCheck & All);
    }

    // Evaluates brdf, returns pdf if oPdf is not NULL
    RT_FUNCTION optix::float3 f( const optix::float3 & aWo,
                                 const optix::float3 & aWi,
                                 float * oPdf = NULL ) const
    {
        if (*oPdf != NULL) *oPdf = 0.f;
        return optix::make_float3(0.0f);
    }

    RT_FUNCTION float pdf( const optix::float3 & aWo, const optix::float3 & aWi, const bool aEvalReverse = false ) const
    {
        return 0.0f;
    }


    RT_FUNCTION optix::float3 sampleF( const optix::float3 & aWo,
                                       optix::float3       * oWi,
                                       const optix::float2 & aSample,
                                       float               * oPdf ) const
    {
        *oWi = localReflect(aWo);
        *oPdf = 0.f;
        return optix::make_float3(0.0f);
    }

    //#define BxDF_rho \
    //__device__ __forceinline__ optix::float3 rho(unsigned int nSamples, \
    //        const float * samples1, const float * samples2) const \
    //{ \
    //    optix::float3 r = optix::make_float3(0.0f); \
    //    for (unsigned int i = 0; i < nSamples; ++i) \
    //    { \
    //        optix::float3 wo, wi; \
    //        wo = sampleUniformSphere(optix::make_float2(samples1[2*i], samples1[2*i+1])); \
    //        float pdf_o = (2.0f * M_PIf), pdf_i = 0.f; \
    //        optix::float3 f = sampleF(wo, &wi, \
    //            optix::make_float2(samples2[2*i], samples2[2*i+1]), &pdf_i); \
    //        if (pdf_i > 0.0f) \
    //            r += f * fabsf(cosTheta(wi)) * fabsf(cosTheta(wo)) / (pdf_o * pdf_i); \
    //    } \
    //    return r / (M_PIf*nSamples); \
    //}


    RT_FUNCTION optix::float3 rho( unsigned int  aNSamples,
                                   const float * aSamples1, 
                                   const float * aSamples2 ) const
    {
        return optix::make_float3(0.0f);
    }

    RT_FUNCTION float reflectProbability() const
    {
        return 0.f;
    }

    RT_FUNCTION float transmitProbability() const
    {
        return 0.f;
    }

    // Evaluation for VCM returning also reverse pdfs, default implementation of "virtual" functions
    RT_FUNCTION optix::float3 vcmF( const optix::float3 & aWo,
                                    const optix::float3 & aWi,
                                    float * oDirectPdf = NULL,
                                    float * oReversePdf = NULL) const
    {
        if (*oDirectPdf != NULL) *oDirectPdf = 0.f;
        if (*oReversePdf != NULL) *oReversePdf = 0.f;
        return optix::make_float3(0.0f);
    }

    RT_FUNCTION void vcmPdf( const optix::float3 & aWo,
                             const optix::float3 & aWi,
                             float * oDirectPdf = NULL,
                             float * oReversePdf = NULL) const
    {
        if (*oDirectPdf != NULL) *oDirectPdf = 0.f;
        if (*oReversePdf != NULL) *oReversePdf = 0.f;
    }

};  /* -----  end of class BxDF  ----- */



class Lambertian : public BxDF 
{
//private:
public:
    optix::float3  _reflectance;

public:
    RT_FUNCTION Lambertian( const optix::float3 & aReflectance ) :
        BxDF(BxDF::Type(BxDF::Lambertian | BxDF::Reflection | BxDF::Diffuse)),
        _reflectance(aReflectance) {  }

    // Evaluates brdf, returns pdf if oPdf is not NULL
    RT_FUNCTION optix::float3 f( const optix::float3 & aWo,
                                 const optix::float3 & aWi,
                                 float * oPdfW = NULL  ) const
    {
        if (*oPdfW != NULL) *oPdfW = pdf(aWo, aWi);
        return _reflectance * M_1_PIf;
    }

    RT_FUNCTION float pdf( const optix::float3 & aWo,
                           const optix::float3 & aWi,
                           const bool aEvalReverse = false ) const
    {
        if (localIsSameHemisphere(aWo, aWi))
        {
            if (!aEvalReverse)
                return fabsf(localCosTheta(aWi)) * M_1_PIf;
            else
                return fabsf(localCosTheta(aWo)) * M_1_PIf;
        }
        return 0.f;
    }

    RT_FUNCTION optix::float3 sampleF( const optix::float3 & aWo,
                                       optix::float3       * oWi,
                                       const optix::float2 & aSample,
                                       float               * oPdfW ) const
    {
        if (aWo.z < EPS_COSINE)
        {
            *oPdfW = 0.f;
            return make_float3(0.f);
        }

        optix::cosine_sample_hemisphere(aSample.x, aSample.y, *oWi);
#if DEBUG_SCATTER_MIRROR_REFLECT // to force repeatable, predictable bounces
        *oWi = localReflect(aWo);
#endif
        *oPdfW = pdf(aWo, *oWi);
        return f(aWo, *oWi);
    }

    RT_FUNCTION optix::float3 rho( unsigned int  aNSamples,
                                   const float * aSamples1,
                                   const float * aSamples2 ) const
    {
        return _reflectance;
    }

    RT_FUNCTION float reflectProbability() const
    {
        return maxf(_reflectance.x, maxf(_reflectance.y, _reflectance.z));
        //return optix::luminanceCIE(_reflectance);  // using luminance causes noise in the image
    }

    RT_FUNCTION float transmitProbability() const
    {
        return 0.f;
    }

    // Evaluation for VCM returning also reverse pdfs, localDirFix in VcmBSDF should be passed as aWo 
    RT_FUNCTION optix::float3 vcmF( const optix::float3 & aWo,
                                    const optix::float3 & aWi,
                                    float * oDirectPdfW = NULL,
                                    float * oReversePdfW = NULL) const
    {
        using namespace optix;
        if(aWo.z < EPS_COSINE || aWi.z < EPS_COSINE)
            return make_float3(0.f);

        if (oDirectPdfW) 
            *oDirectPdfW = fmaxf(0.f, aWi.z * M_1_PIf); // dir.z is equal to cosTheta

        if (oReversePdfW) 
            *oReversePdfW = fmaxf(0.f, aWo.z * M_1_PIf);

        return _reflectance * M_1_PIf;
    }
};


class SpecularReflection : public BxDF {
public:
    RT_FUNCTION SpecularReflection(const optix::float3 & aReflectance, Fresnel * aFresnel) :
        BxDF(BxDF::Type(BxDF::SpecularReflection | BxDF::Reflection | BxDF::Specular)),
        _reflectance(aReflectance) 
    {
        Fresnel *fresnel = reinterpret_cast<Fresnel *>(&_fresnel);
        memcpy(fresnel, aFresnel, MAX_FRESNEL_SIZE); // don't care if copy too much, extra bytes won't be used by target type anyway
    }

public:
    RT_FUNCTION Fresnel * fresnel() 
    {
        return reinterpret_cast<Fresnel *>(_fresnel);
    }

    RT_FUNCTION const Fresnel * fresnel() const
    {
        return reinterpret_cast<const Fresnel *>(_fresnel);
    }

    RT_FUNCTION float reflectProbability() const
    {
        return maxf(_reflectance.x, maxf(_reflectance.y, _reflectance.z));
    }

    RT_FUNCTION float transmitProbability() const
    {
        return 0.f;
    }

    RT_FUNCTION optix::float3 f( const optix::float3 & /* wo */, const optix::float3 & /* wi */, float * oPdfW = NULL) const
    {
        return optix::make_float3(0.0f);
    }

    RT_FUNCTION float pdf(const optix::float3 & wo, const optix::float3 & wi, const bool aEvalReverse = false ) const
    {
        return 0.0f;
    }

    RT_FUNCTION optix::float3 sampleF(
        const optix::float3 & wo, optix::float3 * wi,
        const optix::float2 & sample, float * prob) const
    {
        *wi = optix::make_float3(-wo.x, -wo.y, wo.z);
        *prob = 1.0f;
        optix::float3 F;
        CALL_FRESNEL_CONST_VIRTUAL_FUNCTION(F, =, fresnel(), evaluate, localCosTheta(wo));
        F = F * _reflectance / fabsf(localCosTheta(*wi));
        return F;
    }

private:
    optix::float3  _reflectance;
    char           _fresnel[MAX_FRESNEL_SIZE];
};



class SpecularTransmission : public BxDF 
{
public:
    RT_FUNCTION SpecularTransmission(const optix::float3 & transmittance, float ei, float et) :
        BxDF(BxDF::Type(BxDF::SpecularTransmission | BxDF::Transmission | BxDF::Specular)),
        m_transmittance(transmittance), m_fresnel(ei, et) {  }

public:
    RT_FUNCTION FresnelDielectric * fresnel() { return &m_fresnel; }
    RT_FUNCTION const FresnelDielectric * fresnel() const { return &m_fresnel; }

public:
    RT_FUNCTION optix::float3 f( const optix::float3 & wo , const optix::float3 &  wi, float * oPdfW = NULL ) const
    {
        return optix::make_float3(0.0f);
    }

    RT_FUNCTION float pdf( const optix::float3 & wo, const optix::float3 & wi, const bool aEvalReverse = false ) const
    {
        return 0.0f;
    }

    RT_FUNCTION optix::float3 sampleF(
        const optix::float3 & wo, optix::float3 * wi,
        const optix::float2 & sample, float * prob) const
    {
        using namespace optix;
        
        // Figure out which $\eta$ is incident and which is transmitted
        bool entering = localCosTheta(wo) > 0.0f;
        float ei = fresnel()->eta_i, et = fresnel()->eta_t;
        if (!entering) swap(ei, et);

        // Compute transmitted ray direction
        float sini2 = localSinThetaSquared(wo);
        float eta = ei / et;
        float sint2 = eta * eta * sini2;

        // Handle total internal reflection for transmission
        if (sint2 >= 1.f) return make_float3(0.f);
        float cost = sqrtf(fmaxf(0.f, 1.f - sint2));
        if (entering) cost = -cost;
        float sintOverSini = eta;
        *wi = make_float3(sintOverSini * -wo.x, sintOverSini * -wo.y, cost);
        *prob = 1.f;
        float3 F = fresnel()->evaluate(localCosTheta(wo));
        return (make_float3(1.f) - F) * m_transmittance / fabsf(localCosTheta(*wi));
    }

private:
    optix::float3      m_transmittance;
    FresnelDielectric  m_fresnel;
};


static const unsigned int  MAX_BXDF_SIZE  = sizeof(SpecularTransmission);