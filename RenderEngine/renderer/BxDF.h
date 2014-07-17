// Partially borrowed from https://github.com/LittleCVR/MaoPPM

#pragma once

#include <optixu/optixu_math_namespace.h>
#include "renderer/device_common.h"
#include "renderer/reflection.h"
#include "renderer/helpers/samplers.h"
#include "math/DifferentialGeometry.h"
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
        SpecularTransmission  = 1 << 7,
        Phong                 = 1 << 8
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
                                 float               * oPdf = NULL ) const
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
                                       float               * oPdf,
                                       const bool            aRadianceFromCamera = false ) const
    {
        *oWi = localReflect(aWo);
        *oPdf = 0.f;
        return optix::make_float3(0.0f);
    }

    RT_FUNCTION optix::float3 rho( unsigned int  aNSamples,
                                   const float * aSamples1, 
                                   const float * aSamples2 ) const
    {
        return optix::make_float3(0.0f);
    }

    // for Russian Roulette continuation prob computation
    RT_FUNCTION float continuationProb( const optix::float3 & aWo ) const
    {
        return 0.f;
    }

    // for bxdf sampling probability
    RT_FUNCTION float albedo( const optix::float3 & aWo ) const
    {
        return 0.f;
    }

    // Evaluation for VCM returning also reverse pdfs, default implementation of "virtual" functions
    RT_FUNCTION optix::float3 vcmF( const optix::float3 & aWo,
                                    const optix::float3 & aWi,
                                    float               * oDirectPdf = NULL,
                                    float               * oReversePdf = NULL ) const
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
                           const bool            aEvalReverse = false ) const
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
                                       float               * oPdfW,
                                       const bool            aRadianceFromCamera = false ) const
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

    // for Russian Roulette continuation prob computation
    RT_FUNCTION float continuationProb( const optix::float3 & aWo ) const
    {
        return optix::fmaxf(_reflectance.x, maxf(_reflectance.y, _reflectance.z));
    }

    // for bxdf sampling probability
    RT_FUNCTION float albedo( const optix::float3 & aWo ) const
    {
        return optix::luminanceCIE(_reflectance);
    }

    // Evaluation for VCM returning also reverse pdfs, localDirFix in VcmBSDF should be passed as aWo 
    RT_FUNCTION optix::float3 vcmF( const optix::float3 & aWo,
                                    const optix::float3 & aWi,
                                    float               * oDirectPdfW = NULL,
                                    float               * oReversePdfW = NULL) const
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



#define EPS_PHONG 1e-3f

class Phong : public BxDF 
{
    //private:
public:
    optix::float3  _reflectance;
    float          _exponent;

public:
    RT_FUNCTION Phong( const optix::float3 & aReflectance, const float aExponent ) :
        BxDF(BxDF::Type( BxDF::Phong | BxDF::Reflection | BxDF::Glossy )),
        _reflectance(aReflectance), _exponent(aExponent) {  }

    // for Russian Roulette continuation prob computation
    RT_FUNCTION float continuationProb( const optix::float3 & aWo ) const
    {
        return optix::fmaxf(_reflectance.x, maxf(_reflectance.y, _reflectance.z));
    }

    // for bxdf sampling probability
    RT_FUNCTION float albedo( const optix::float3 & aWo ) const
    {
        return optix::luminanceCIE(_reflectance);
    }

    // Evaluates brdf, returns pdf if oPdf is not NULL
    RT_FUNCTION optix::float3 f( const optix::float3 & aWo,
                                 const optix::float3 & aWi,
                                 float * oPdfW = NULL  ) const
    {
        using namespace optix;

        const float3 reflLocalDirIn = localReflect(aWo);
        const float dot_R_Wi = dot(reflLocalDirIn, aWi);

        if (dot_R_Wi <= EPS_PHONG)
            return make_float3(0.f);

        if (oPdfW)
            *oPdfW = powerCosHemispherePdfW(reflLocalDirIn, aWi, _exponent);

        float3 rho = _reflectance * (_exponent + 2.f) * 0.5f * M_1_PIf;
        return rho * powf(dot_R_Wi, _exponent);
    }

    RT_FUNCTION float pdf( const optix::float3 & aWo, // dirFix for VCN
                           const optix::float3 & aWi, // dirGen
                           const bool            aEvalReverse = false ) const
    {
        using namespace optix;
        const float3 reflLocalDirIn = localReflect(aWo);
        const float dot_R_Wi = dot(reflLocalDirIn, aWi);

        if (dot_R_Wi <= EPS_PHONG)
            return 0.f;

        return powerCosHemispherePdfW(reflLocalDirIn, aWi, _exponent);
    }

    RT_FUNCTION optix::float3 sampleF( const optix::float3 & aWo,
                                       optix::float3       * oWi,
                                       const optix::float2 & aSample,
                                       float               * oPdfW,
                                       const bool            aRadianceFromCamera = false ) const
    {
        using namespace optix;

        *oWi = samplePowerCosHemisphereW(aSample, _exponent, NULL);
        
        // Comment from SmallVCN:
        // Due to numeric issues in MIS, we actually need to compute all pdfs
        // exactly the same way all the time!!!
        const float3 reflLocalDirIn = localReflect(aWo);
        {
            DifferentialGeometry dg;
            dg.SetFromNormal(reflLocalDirIn);
            *oWi = dg.ToWorld(*oWi);   // sampled dir transformed to local frame
        }

        const float dot_R_Wi = dot(reflLocalDirIn, *oWi);
        
        if (dot_R_Wi <= EPS_PHONG)
            return make_float3(0.f);

        if (oPdfW)
            *oPdfW = pdf(aWo, *oWi);

        float3 rho = _reflectance * (_exponent + 2.f) * 0.5f * M_1_PIf;
        return rho * powf(dot_R_Wi, _exponent);
    }

    RT_FUNCTION optix::float3 rho( unsigned int  aNSamples,
                                   const float * aSamples1,
                                   const float * aSamples2 ) const
    {
        return _reflectance * (_exponent + 2.f) * 0.5f * M_1_PIf;
    }


    // Evaluation for VCM returning also reverse pdfs, localDirFix in VcmBSDF should be passed as aWo 
    RT_FUNCTION optix::float3 vcmF( const optix::float3 & aWo,
                                    const optix::float3 & aWi,
                                    float               * oDirectPdfW = NULL,
                                    float               * oReversePdfW = NULL ) const
    {
        using namespace optix;
        float pdf;
        float3 f = this->f(aWo, aWi, &pdf);        
        if (oDirectPdfW)  *oDirectPdfW = pdf;
        if (oReversePdfW)  *oReversePdfW = pdf;
        return f;
    }
};



class SpecularReflection : public BxDF 
{
private:
    optix::float3  _reflectance;
    char           _fresnel[MAX_FRESNEL_SIZE];

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

    // for Russian Roulette continuation prob computation
    RT_FUNCTION float continuationProb( const optix::float3 & aWo ) const
    {
        float R;
        CALL_FRESNEL_CONST_VIRTUAL_FUNCTION(R, =, fresnel(), evaluate, localCosTheta(aWo));
        return R * optix::fmaxf(_reflectance.x, maxf(_reflectance.y, _reflectance.z));
    }

    // for bxdf sampling probability
    RT_FUNCTION float albedo( const optix::float3 & aWo ) const
    {
        float R;
        CALL_FRESNEL_CONST_VIRTUAL_FUNCTION(R, =, fresnel(), evaluate, localCosTheta(aWo));
        return R * optix::luminanceCIE(_reflectance);
    }

    RT_FUNCTION optix::float3 f( const optix::float3 & /* wo */, const optix::float3 & /* wi */, float * oPdfW = NULL) const
    {
        return optix::make_float3(0.0f);
    }

    RT_FUNCTION float pdf(const optix::float3 & wo, const optix::float3 & wi, const bool aEvalReverse = false ) const
    {
        return 0.0f;
    }

    RT_FUNCTION optix::float3 sampleF( const optix::float3 & aWo,
                                       optix::float3       * oWi,
                                       const optix::float2 & aSample,
                                       float               * oPdfW,
                                       const bool            aRadianceFromCamera = false ) const
    {
        *oWi = optix::make_float3(-aWo.x, -aWo.y, aWo.z);
        *oPdfW = 1.0f;
        float R;
        CALL_FRESNEL_CONST_VIRTUAL_FUNCTION(R, =, fresnel(), evaluate, localCosTheta(aWo));

        // BSDF is multiplied by cosThetaOut when computing throughput to scattered direction. It shouldn't
        // be done for specular reflection, hence predivide here to cancel it out
        return R * _reflectance / fabsf(localCosTheta(*oWi));
    }
};




class SpecularTransmission : public BxDF 
{
private:
    optix::float3      _transmittance;
    FresnelDielectric  _fresnel;

public:
    RT_FUNCTION SpecularTransmission(const optix::float3 & transmittance, float ei, float et) :
        BxDF(BxDF::Type(BxDF::SpecularTransmission | BxDF::Transmission | BxDF::Specular)),
        _transmittance(transmittance), _fresnel(ei, et) {  }

public:
    RT_FUNCTION FresnelDielectric * fresnel() { return &_fresnel; }
    RT_FUNCTION const FresnelDielectric * fresnel() const { return &_fresnel; }

public:
    RT_FUNCTION optix::float3 f( const optix::float3 & wo , const optix::float3 &  wi, float * oPdfW = NULL ) const
    {
        return optix::make_float3(0.0f);
    }

    RT_FUNCTION float pdf( const optix::float3 & wo, const optix::float3 & wi, const bool aEvalReverse = false ) const
    {
        return 0.0f;
    }

    // for Russian Roulette continuation prob computation
    RT_FUNCTION float continuationProb( const optix::float3 & aWo ) const
    {
        float R;
        CALL_FRESNEL_CONST_VIRTUAL_FUNCTION(R, =, fresnel(), evaluate, localCosTheta(aWo));
        return 1.f - R;
    }

    // for bxdf sampling probability
    RT_FUNCTION float albedo( const optix::float3 & aWo ) const
    {
        float R;
        CALL_FRESNEL_CONST_VIRTUAL_FUNCTION(R, =, fresnel(), evaluate, localCosTheta(aWo));
        return (1.f - R) * optix::luminanceCIE(_transmittance);
    }

    RT_FUNCTION optix::float3 sampleF( const optix::float3 & aWo,
                                       optix::float3       * oWi,
                                       const optix::float2 & aSample,
                                       float               * oPdfW,
                                       const bool            aRadianceFromCamera = false ) const
    {
        using namespace optix;
        
        // Figure out which eta is incident and which is transmitted
        bool entering = localCosTheta(aWo) > 0.0f;
        float ei = fresnel()->eta_i, et = fresnel()->eta_t;
        if (!entering) swap(ei, et);

        // Compute transmitted ray direction
        float sini2 = localSinThetaSquared(aWo);
        float eta = ei / et;
        float sint2 = eta * eta * sini2;

        // Handle total internal reflection for transmission
        if (sint2 >= 1.f) return make_float3(0.f);
        float cost = sqrtf(fmaxf(0.f, 1.f - sint2));
        if (entering) cost = -cost;
        float sintOverSini = eta;
        *oWi = make_float3(sintOverSini * -aWo.x, sintOverSini * -aWo.y, cost);
        *oPdfW = 1.f;

        // reflection/refraction coefficients
        float R = fresnel()->evaluate(localCosTheta(aWo));
        float T = 1.f - R;

        // aRadianceFromCamera used for VCM when radiance flows from camera and particle importance/weights from light.
        // etas are swapped, hence the scaling
        if (aRadianceFromCamera)
            return /*(ei*ei)/(et*et) * */ T * _transmittance * sqr(sintOverSini) / fabsf(localCosTheta(*oWi));
        else
            return /*(ei*ei)/(et*et) * */ T * _transmittance / fabsf(localCosTheta(*oWi));
    }
};


static const unsigned int  MAX_BXDF_SIZE  = sizeof(SpecularReflection);