#pragma once

// Partially borrowed from https://github.com/LittleCVR/MaoPPM

#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include "renderer/helpers/reflection.h"
#include "BxDF.h"
#include "math/DifferentialGeometry.h"


#define CALL_BXDF_CONST_VIRTUAL_FUNCTION(lvalue, op, bxdf, function, ...) \
    if (bxdf->type() & BxDF::Lambertian) \
        lvalue op reinterpret_cast<const Lambertian *>(bxdf)->function(__VA_ARGS__);

class BSDF 
{
public:
    static const unsigned int  MAX_N_BXDFS  = 2;

protected:
    DifferentialGeometry  _diffGemetry;
    optix::float3         _geometricNormal;
    unsigned int          _nBxDFs;
    char                  _bxdfList [MAX_N_BXDFS * MAX_BXDF_SIZE];
    float                 _bxdfPickProb [MAX_N_BXDFS]; // unscaled component continuation probabilities
    float                 _continuationProb;           // Russian roulette probability

public:
    __device__ __forceinline__ BSDF() {  }
    __device__ __forceinline__ BSDF( const DifferentialGeometry aDiffGeomShading,
                                     const optix::float3      & aWorldGeometricNormal )
    {
        _geometricNormal = aWorldGeometricNormal;
        _diffGemetry = aDiffGeomShading;
        _nBxDFs = 0;
    }

    // For simple case when not differentiating between gemoetric and shading normal,
    // generates tangent and bitangent
    __device__ __forceinline__ BSDF( const optix::float3 & aWorldNormal )
    {
        _diffGemetry.SetFromZ(aWorldNormal);
        _geometricNormal = aWorldNormal;
        _nBxDFs = 0;
    }

    __device__ __forceinline__ unsigned int nBxDFs() const { return _nBxDFs; }

    __device__ __forceinline__ unsigned int nBxDFs(BxDF::Type aType) const
    {
        unsigned int count = 0;
        for (unsigned int i = 0; i < _nBxDFs; ++i)
        {
            if (bxdfAt(i)->matchFlags(aType))
                ++count;
        }
        return count;
    }

    __device__ __forceinline__ float continuationProb() const { return _continuationProb; }

    // Add BxDF. Returns 0 if failed, e.g. MAX_N_BXDFS reached
    __device__ __inline__ int AddBxdf(const BxDF & bxdf)
    {
        if (_nBxDFs == MAX_N_BXDFS) return 0;

        BxDF *pBxDF = reinterpret_cast<BxDF *>(&_bxdfList[_nBxDFs * MAX_BXDF_SIZE]);
        *pBxDF = bxdf;

        float pickProb = pBxDF->reflectProbability() + pBxDF->transmitProbability();
        _bxdfPickProb[_nBxDFs] = pickProb;

        // Setting continuation probability explicitly (instead of using arbitrary values) for russian roulette 
        // to make sure the weight of sample never rise
        _continuationProb = optix::fminf(1.f, _continuationProb + pickProb);

        _nBxDFs++;
        return 1;
    }

    __device__ __inline__ const BxDF * bxdfAt(const optix::uint & aIndex) const
    {
        return reinterpret_cast<const BxDF *>(&_bxdfList[aIndex * MAX_BXDF_SIZE]);
    }

    __device__ __inline__ const BxDF * bxdfAt(const optix::uint & aIndex, BxDF::Type aType) const
    {
        optix::uint count = aIndex;
        for (unsigned int i = 0; i < nBxDFs(); ++i) 
        {
            if (bxdfAt(i)->matchFlags(aType))
            {
                if (count != 0)
                    --count;
                else
                    return bxdfAt(i);
            }
        }
        return NULL;
    }

    __device__ __forceinline__ bool isSpecular() const
    {
        return (nBxDFs(BxDF::Type(BxDF::All & ~BxDF::Specular)) == 0);
    }

    // Return bsdf factor for directions oWorldWi and aWorldWi
    // Following typical conventions Wo corresponds to light outgoing direction, 
    // Wi is incident direction. Returns pdf if oPdf not NULL
    __device__ optix::float3 f( const optix::float3 & aWorldWo,
                                const optix::float3 & aWorldWi, 
                                BxDF::Type            aSampleType = BxDF::All,
                                float               * oPdf = NULL) const
    {
        optix::float3 wo = _diffGemetry.ToLocal(aWorldWo);
        optix::float3 wi = _diffGemetry.ToLocal(aWorldWi);

        // Calculate f.
        if (optix::dot(_geometricNormal, aWorldWi) * optix::dot(_geometricNormal, aWorldWo) >= 0.0f)  
            aSampleType = BxDF::Type(aSampleType & ~BxDF::Transmission);      // ignore BTDF
        else
            aSampleType = BxDF::Type(aSampleType & ~BxDF::Reflection);        // ignore BRDF
        
        // Sum all matched BxDF's f and probability
        optix::float3 f = optix::make_float3(0.0f);
        float tPdf;
        int numMatched = 0;

        for (unsigned int i = 0; i < _nBxDFs; ++i)
        {
            if (bxdfAt(i)->matchFlags(aSampleType))
            {
                tPdf = 0.f;
                CALL_BXDF_CONST_VIRTUAL_FUNCTION(f, +=, bxdfAt(i), f, wo, wi, &tPdf);
                if (oPdf) *oPdf += tPdf;
                numMatched++;
            }
        }

        if (oPdf && 1 < numMatched)
            *oPdf /= static_cast<float>(numMatched);

        return f;
    }


    // Return bsdf factor for sampled direction oWorldWi. Returns pdf in and sampled BxDF.
    // Following typical conventions Wo corresponds to light outgoing direction, 
    // Wi is sampled incident direction
    __device__ optix::float3 sampleF( const optix::float3 & aWorldWo,
                                      optix::float3       * oWorldWi, 
                                      const optix::float3 & aSample,
                                      float               * oPdfW,
                                      BxDF::Type            aSampleType = BxDF::All,
                                      BxDF::Type          * oSampledType = NULL ) const
    {
        // Count matched componets.
        //unsigned int nMatched = nBxDFs(aSampleType);
        //if (nMatched == 0)
        //{
        //    *oPdfW = 0.0f;
        //    if (oSampledType) *oSampledType = BxDF::Null;
        //    return optix::make_float3(0.0f);
        //}

        //float matchedSumContProb = sumContProb(aSampleType);

        // vmarz TODO pick based on albedo of each component as in SmallVCM?
        // Sample BxDF.
        //unsigned int index = optix::min(nMatched-1,
        //        static_cast<unsigned int>(floorf(aSmple.x * static_cast<float>(nMatched))));
        unsigned int index = 0;
        unsigned int nMatched = sampleBxDF(aSample.x, aSampleType, &index);

        const BxDF * bxdf = bxdfAt(index, aSampleType);
        if (bxdf == NULL)
        {
            *oPdfW = 0.0f;
            if (oSampledType) *oSampledType = BxDF::Null;
            return optix::make_float3(0.0f);
        }

        // Transform.
        optix::float3 wo = _diffGemetry.ToLocal(aWorldWo);

        // Sample f.
        optix::float3 f;
        optix::float3 wi;
        optix::float2 s = optix::make_float2(aSample.y, aSample.z);
        CALL_BXDF_CONST_VIRTUAL_FUNCTION(f, =, bxdf, sampleF, wo, &wi, s, oPdfW);
        
        // Rejected.
        if (*oPdfW == 0.0f)
        {
            if (oSampledType) *oSampledType = BxDF::Null;
            return optix::make_float3(0.0f);
        }

        // Otherwise.
        if (oSampledType) *oSampledType = bxdf->type();
        *oWorldWi = _diffGemetry.ToWorld(wi);

        // If not specular, sum all non-specular BxDF's probability.
        if (!(bxdf->type() & BxDF::Specular) && nMatched > 1) 
        {
            //*oPdf = 1.0f; // original - vmarz why this ?
            for (unsigned int i = 0; i < _nBxDFs; i++) // vmarz original condition was i<1
            {
                if (i == index) continue; // index of bxdf used for direction sampling, skip computing pdf
                if (bxdfAt(i)->matchFlags(aSampleType))
                    CALL_BXDF_CONST_VIRTUAL_FUNCTION(*oPdfW, +=, bxdfAt(i), pdf, wo, wi);
            }
        }
            
        // Remember to divide component count.
        if (nMatched > 1)
            *oPdfW /= static_cast<float>(nMatched);
            
        // If not specular, sum all f.
        if (!(bxdf->type() & BxDF::Specular))
        {
            // f = make_float3(0.0f); // vmarz commented out. reuse value computed when direction was 

            // Cannot use localIsSameHemisphere(wo, *wi) here,
            // do not confuse with the geometric normal and the shading normal.
            if (optix::dot(_geometricNormal, *oWorldWi) * optix::dot(_geometricNormal, aWorldWo) >= 0.0f) 
                aSampleType = BxDF::Type(aSampleType & ~BxDF::Transmission);      // ignore BTDF
            else                                                                 
                aSampleType = BxDF::Type(aSampleType & ~BxDF::Reflection);        // ignore BRDF
                
            for (unsigned int i = 0; i < _nBxDFs; ++i)
            {
                if (i == index) continue; // index of bxdf used for direction sampling, skip computing f again
                if (bxdfAt(i)->matchFlags(aSampleType))
                    CALL_BXDF_CONST_VIRTUAL_FUNCTION(f, +=, bxdfAt(i), f, wo, wi);
            }
        }

        return f;
    }

protected:
    __device__ __forceinline__ unsigned int sampleBxDF(float sample, BxDF::Type aType, unsigned int * bxdfIndex) const
    {
        unsigned int nMatched = nBxDFs(aType);
        if (nMatched == 0) return 0;

        float matchedSumContProb = sumContProb(aType);
        float contProbPrev = 0.f;
        float contProb = 0.f;
        for (unsigned int i = 0; i < _nBxDFs; ++i)
        {
            if (bxdfAt(i)->matchFlags(aType))
            {
                float contProb = _bxdfPickProb[i] / matchedSumContProb;
                if (sample < contProbPrev + contProb)
                {
                    *bxdfIndex = i;
                    break;
                }
                contProbPrev += contProb;
            }
        }

        return nMatched;
    }


    __device__ __forceinline__ float sumContProb(BxDF::Type aType) const
    {
        float contProb = 0.f;
        for (unsigned int i = 0; i < _nBxDFs; ++i)
        {
            if (bxdfAt(i)->matchFlags(aType))
            {
                contProb += _bxdfPickProb[i];
            }
        }
        return contProb;
    }

};  /* -----  end of class BSDF  ----- */



class VcmBSDF : public BSDF
{
private:
    optix::float3 _localDirFix;    // following convention in SmallVCM, "fix" is corresponds to incident dir at hit point
                                   // opposed to "gen" for generated
public:
    __device__ __forceinline__ VcmBSDF() : BSDF() { }

    __device__ __forceinline__ VcmBSDF( const DifferentialGeometry aDiffGeomShading,
                                        const optix::float3      & aWorldGeometricNormal,
                                        const optix::float3      & aIncidentDir ) : BSDF(aWorldGeometricNormal)
    {
        _localDirFix = _diffGemetry.ToLocal(aIncidentDir);
    }

    // For simple case when not differentiating between gemoetric and shading normal,
    // generates tangent and bitangent
    __device__ __forceinline__ VcmBSDF( const optix::float3 & aWorldNormal,
                                        const optix::float3 & aIncidentDir ) : BSDF(aWorldNormal)
    {
        _localDirFix = _diffGemetry.ToLocal(aIncidentDir);
    }

    __device__ __forceinline__ optix::float3 localDirFix() const { return _localDirFix; }


    // Estimates bsdf factor for directions oWorldWi and aWorldWi and pdfs.
    // In typical conventions Wo corresponds to light outgoing direction, Wi to generated incident direction.
    // For VCM evaluation the stored direction localDirFix is used as Wo, generated direction aWorldDirGen as Wi,
    // either when tracing from light or camera. Similary directPdf corresponds sampling from Wo->Wi, reverse to Wi->Wo
    __device__ optix::float3 vcmF( const optix::float3 & aWorldDirGen,
                                   float               * oDirectPdfW = NULL,
                                   float               * oReversePdfW = NULL,
                                   BxDF::Type            aSampleType = BxDF::All ) const
    {
        optix::float3 localDirGen = _diffGemetry.ToLocal(aWorldDirGen);
        optix::float3 worldDirFix = _diffGemetry.ToWorld(_localDirFix);

        if (oDirectPdfW) *oDirectPdfW = 0.f;
        if (oReversePdfW) *oReversePdfW = 0.f;

        if (_localDirFix.z < EPS_COSINE || localDirGen.z < EPS_COSINE)
            return make_float3(0.f);

        // Calculate f.
        if (optix::dot(_geometricNormal, aWorldDirGen) * optix::dot(_geometricNormal, worldDirFix) >= 0.0f)  
            aSampleType = BxDF::Type(aSampleType & ~BxDF::Transmission);      // ignore BTDF
        else
            aSampleType = BxDF::Type(aSampleType & ~BxDF::Reflection);        // ignore BRDF
        
        // Sum all matched BxDF's f and probability
        optix::float3 f = optix::make_float3(0.0f);
        float dPdfW, rPdfW;
        int numMatched = 0;

        for (unsigned int i = 0; i < _nBxDFs; ++i)
        {
            if (bxdfAt(i)->matchFlags(aSampleType))
            {
                dPdfW = rPdfW = 0.f;
                CALL_BXDF_CONST_VIRTUAL_FUNCTION(f, +=, bxdfAt(i), vcmF, _localDirFix, localDirGen, &dPdfW, &rPdfW);
                if (oDirectPdfW) *oDirectPdfW += dPdfW;
                if (oReversePdfW) *oReversePdfW += rPdfW;
                numMatched++;
            }
        }

        if (1 < numMatched)
        {
            if (oDirectPdfW) *oDirectPdfW /= static_cast<float>(numMatched);
            if (oReversePdfW) *oReversePdfW /= static_cast<float>(numMatched);
        }
        return f;
    }

};

