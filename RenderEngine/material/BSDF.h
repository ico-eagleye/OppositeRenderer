#pragma once

// Partially borrowed from https://github.com/LittleCVR/MaoPPM

#include  <optix_world.h>
#include "renderer/helpers/reflection.h"
#include "BxDF.h"
#include "math/DifferentialGeometry.h"


#define CALL_BXDF_CONST_VIRTUAL_FUNCTION(lvalue, op, bxdf, function, ...) \
    if (bxdf->type() & BxDF::Lambertian) \
        lvalue op reinterpret_cast<const Lambertian *>(bxdf)->function(__VA_ARGS__);

class BSDF {
    public:
        static const unsigned int  MAX_N_BXDFS  = 2;
    private:
        // should be private
        DifferentialGeometry  _diffGemetry;
        optix::float3         _geometricNormal;
        optix::float3         _localDirFix;    // following convention in SmallVCM, "fix" is corresponds to incident dir at hit point
                                               // opposed to "gen" for generated
        unsigned int          _nBxDFs;
        char                  _bxdfList [MAX_N_BXDFS * MAX_BXDF_SIZE];

//#ifdef __CUDACC__
    public:
        __device__ __forceinline__ BSDF() {  }
        __device__ __forceinline__ BSDF( const DifferentialGeometry aDiffGeomShading,
                                         const optix::float3      & aGeometricNormal,
                                         const optix::float3      & aIncidentDir)
        {
            _geometricNormal = aGeometricNormal;
            _diffGemetry = aDiffGeomShading;
            _localDirFix = _diffGemetry.ToLocal(aIncidentDir);
            _nBxDFs = 0;
        }

        // For simple case when not differentiating between gemoetric and shading normal,
        // generates tangent and bitangent
        __device__ __forceinline__ BSDF( const optix::float3 & aNormal,
                                         const optix::float3 & aIncidentDir )
        {
            _diffGemetry.SetFromZ(aNormal);
            _geometricNormal = aNormal;
            _localDirFix = _diffGemetry.ToLocal(aIncidentDir);
            _nBxDFs = 0;
        }

        __device__ __forceinline__ optix::float3 localDirFix() const { return _localDirFix; }

        __device__ __forceinline__ unsigned int nBxDFs() const { return _nBxDFs; }

        __device__ __forceinline__ unsigned int nBxDFs(BxDF::Type type) const
        {
            unsigned int count = 0;
            for (unsigned int i = 0; i < nBxDFs(); ++i)
            {
                if (bxdfAt(i)->matchFlags(type))
                    ++count;
            }
            return count;
        }

        __device__ __inline__ BxDF * bxdfAt(const optix::uint & aIndex)
        {
            const BSDF * bsdf = this;
            return const_cast<BxDF *>(bsdf->bxdfAt(aIndex));
        }

        __device__ __inline__ const BxDF * bxdfAt(const optix::uint & aIndex) const
        {
            return reinterpret_cast<const BxDF *>(&_bxdfList[aIndex * MAX_BXDF_SIZE]);
        }

        __device__ __inline__ BxDF * bxdfAt(const optix::uint & aIndex, BxDF::Type aType)
        {
            const BSDF * bsdf = this;
            return const_cast<BxDF *>(bsdf->bxdfAt(aIndex, aType));
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

    public:
        __device__ optix::float3 f(const optix::float3 & aWorldWo,
                                   const optix::float3 & aWorldWi, 
                                   BxDF::Type            aSampleType = BxDF::All) const
        {
            optix::float3 wo = _diffGemetry.ToLocal(aWorldWo);
            optix::float3 wi = _diffGemetry.ToLocal(aWorldWi);

            // Calculate f.
            optix::float3 f = optix::make_float3(0.0f);
            if (optix::dot(_geometricNormal, aWorldWi) * optix::dot(_geometricNormal, aWorldWo) >= 0.0f)  
                aSampleType = BxDF::Type(aSampleType & ~BxDF::Transmission);      // ignore BTDF
            else
                aSampleType = BxDF::Type(aSampleType & ~BxDF::Reflection);        // ignore BRDF
            
            for (unsigned int i = 0; i < nBxDFs(); ++i)
            {
                if (bxdfAt(i)->matchFlags(aSampleType))
                    CALL_BXDF_CONST_VIRTUAL_FUNCTION(f, +=, bxdfAt(i), f, wo, wi);
            }
            return f;
        }



        __device__ optix::float3 sampleF(const optix::float3 & aWorldWo,
                                         optix::float3       * oWorldWi, 
                                         const optix::float3 & aSmple,
                                         float               * oProb,
                                         BxDF::Type            aSampleType = BxDF::All,
                                         BxDF::Type          * oSampledType = NULL) const
        {
            // Count matched componets.
            unsigned int nMatched = nBxDFs(aSampleType);
            if (nMatched == 0)
            {
                *oProb = 0.0f;
                if (oSampledType) *oSampledType = BxDF::Null;
                return optix::make_float3(0.0f);
            }

            // vmarz TODO pick based on albedo as in SmallVCM?
            // Sample BxDF.
            unsigned int index = intmin(nMatched-1,
                    static_cast<unsigned int>(floorf(aSmple.x * static_cast<float>(nMatched))));
            const BxDF * bxdf = bxdfAt(index, aSampleType);
            if (bxdf == NULL)
            {
                *oProb = 0.0f;
                if (oSampledType) *oSampledType = BxDF::Null;
                return optix::make_float3(0.0f);
            }

            // Transform.
            optix::float3 wo = _diffGemetry.ToLocal(aWorldWo);

            // Sample f.
            optix::float3 f;
            optix::float3 wi;
            optix::float2 s = optix::make_float2(aSmple.y, aSmple.z);
            CALL_BXDF_CONST_VIRTUAL_FUNCTION(f, =, bxdf, sampleF, wo, &wi, s, oProb);
            // Rejected.
            if (*oProb == 0.0f)
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
                *oProb = 1.0f;
                for (unsigned int i = 0; i < 1; i++)
                {
                    if (bxdfAt(i)->matchFlags(aSampleType))
                        CALL_BXDF_CONST_VIRTUAL_FUNCTION(*oProb, +=, bxdfAt(i), pdf, wo, wi);
                }
            }
            
            // Remember to divide component count.
            if (nMatched > 1)
                *oProb /= static_cast<float>(nMatched);
            
            // If not specular, sum all f.
            if (!(bxdf->type() & BxDF::Specular))
            {
                f = make_float3(0.0f);
                // Cannot use localIsSameHemisphere(wo, *wi) here,
                // do not confuse with the geometric normal and the shading normal.
                if (optix::dot(_geometricNormal, *oWorldWi) * optix::dot(_geometricNormal, aWorldWo) >= 0.0f) 
                    aSampleType = BxDF::Type(aSampleType & ~BxDF::Transmission);      // ignore BTDF
                else                                                                 
                    aSampleType = BxDF::Type(aSampleType & ~BxDF::Reflection);        // ignore BRDF
                
                for (unsigned int i = 0; i < nBxDFs(); ++i)
                {
                    if (bxdfAt(i)->matchFlags(aSampleType))
                        CALL_BXDF_CONST_VIRTUAL_FUNCTION(f, +=, bxdfAt(i), f, wo, wi);
                }
            }

            return f;
        }
//#endif  /* -----  #ifdef __CUDACC__  ----- */

};  /* -----  end of class BSDF  ----- */


