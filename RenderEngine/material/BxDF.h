#pragma once

// Partially borrowed from https://github.com/LittleCVR/MaoPPM

#include <optixu/optixu_math_namespace.h>
#include "renderer/helpers/reflection.h"
#include "renderer/helpers/samplers.h"
#include "config.h"

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
    };

private:
    Type  _type;

public:
    __device__ __forceinline__ BxDF(Type type) : _type(type) {  }

    __device__ __forceinline__ Type type() const { return _type; }

    // Because we compress the basic types and BxDF types in a single
    // $_type variable, it is necessary to AND All first.
    __device__ __forceinline__ bool matchFlags(Type type) const
    {
        return (_type & All & type) == (_type & All);
    }

    // Evaluates brdf, returns pdf if oPdf is not NULL
    __device__ __forceinline__ optix::float3 f( const optix::float3 & aWo,
                                                const optix::float3 & aWi,
                                                float * oPdf = NULL ) const
    {
        if (*oPdf != NULL) *oPdf = 0.f;
        return optix::make_float3(0.0f);
    }

    __device__ __forceinline__ float pdf( const optix::float3 & aWo,
                                          const optix::float3 & aWi ) const
    {
        return localIsSameHemisphere(aWo, aWi) ? fabsf(localCosTheta(aWi)) * M_1_PIf : 0.0f;
    }


    __device__ __forceinline__ optix::float3 sampleF( const optix::float3 & aWo,
                                                      optix::float3       * oWi,
                                                      const optix::float2 & aSample,
                                                      float               * oPdf ) const
    {
        optix::cosine_sample_hemisphere(aSample.x, aSample.y, *oWi);
        *oPdf = pdf(aWo, *oWi);
        return f(aWo, *oWi);
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


    __device__ __forceinline__ optix::float3 rho( unsigned int  aNSamples,
                                                  const float * aSamples1, 
                                                  const float * aSamples2 ) const
    {
        return optix::make_float3(0.0f);
    }

    // Evaluation for VCM returning also reverse pdfs
    __device__ __forceinline__ optix::float3 vcmF( const optix::float3 & aWo,
                                                   const optix::float3 & aWi,
                                                   float * oDirectPdf = NULL,
                                                   float * oReversePdf = NULL) const
    {
        if (*oDirectPdf != NULL) *oDirectPdf = 0.f;
        if (*oReversePdf != NULL) *oReversePdf = 0.f;
        return optix::make_float3(0.0f);
    }

};  /* -----  end of class BxDF  ----- */



class Lambertian : public BxDF 
{
private:
    optix::float3  _reflectance;

public:
    __device__ __forceinline__ Lambertian( const optix::float3 & aReflectance ) :
        BxDF(BxDF::Type(BxDF::Lambertian | BxDF::Reflection | BxDF::Diffuse)),
        _reflectance(aReflectance) {  }

    // Evaluates brdf, returns pdf if oPdf is not NULL
    __device__ __forceinline__ optix::float3 f( const optix::float3 & aWo,
                                                const optix::float3 & aWi,
                                                float * oPdfW = NULL  ) const
    {
        if (*oPdfW != NULL) *oPdfW = pdf(aWo, aWi);
        return _reflectance * M_1_PIf;
    }

    __device__ __forceinline__ float pdf( const optix::float3 & aWo,
                                          const optix::float3 & aWi ) const
    {
        return localIsSameHemisphere(aWo, aWi) ? fabsf(localCosTheta(aWi)) * M_1_PIf : 0.0f;
    }


    __device__ __forceinline__ optix::float3 sampleF( const optix::float3 & aWo,
                                                      optix::float3       * oWi,
                                                      const optix::float2 & aSample,
                                                      float               * oPdfW ) const
    {
        optix::cosine_sample_hemisphere(aSample.x, aSample.y, *oWi);
        *oPdfW = pdf(aWo, *oWi);
        return f(aWo, *oWi);
    }

    __device__ __forceinline__ optix::float3 rho( unsigned int  aNSamples,
                                                  const float * aSamples1,
                                                  const float * aSamples2 ) const
    {
        return _reflectance;
    }

    // Evaluation for VCM returning also reverse pdfs, localDirFix in VcmBSDF should be passed as aWo 
    __device__ __forceinline__ optix::float3 vcmF( const optix::float3 & aWo,
                                                   const optix::float3 & aWi,
                                                   float * oDirectPdfW = NULL,
                                                   float * oReversePdfW = NULL) const
    {
        if(aWo.z < EPS_COSINE || aWi.z < EPS_COSINE)
            return make_float3(0.f);

        if (*oDirectPdfW != NULL) 
            *oDirectPdfW = optix::fmaxf(0.f, aWi.z * M_1_PIf); // dir.z is equal to cosTheta

        if (*oReversePdfW != NULL) 
            *oReversePdfW =  optix::fmaxf(0.f, aWo.z * M_1_PIf);
        return _reflectance * M_1_PIf;
    }
};  /* -----  end of class Lambertian  ----- */


static const unsigned int  MAX_BXDF_SIZE  = sizeof(Lambertian);