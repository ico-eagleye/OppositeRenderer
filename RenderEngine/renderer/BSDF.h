// Partially borrowed from https://github.com/LittleCVR/MaoPPM

#pragma once

//#define OPTIX_PRINTF_DEF
//#define OPTIX_PRINTFI_DEF
//#define OPTIX_PRINTFID_DEF

#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include "renderer/device_common.h"
#include "renderer/reflection.h"
#include "BxDF.h"
#include "math/DifferentialGeometry.h"
#include <device_functions.h>
#include "renderer/vcm/config_vcm.h"

// having printfs enabled here sometimes cause weird errors like "insn->isMove() || insn->isLoad() || insn->isAdd()"
// have seen that with inlining problems
#define OPTIX_PRINTF_ENABLED 0
#define OPTIX_PRINTFI_ENABLED 0
#define OPTIX_PRINTFID_ENABLED 0
#define OPTIX_PRINTFC_ENABLED 0

#define CALL_BXDF_CONST_VIRTUAL_FUNCTION(lvalue, op, bxdf, function, ...) \
    if (bxdf->type() & BxDF::Lambertian) \
        lvalue op reinterpret_cast<const Lambertian *>(bxdf)->function(__VA_ARGS__); \
    else if (bxdf->type() & BxDF::SpecularReflection) \
        lvalue op reinterpret_cast<const SpecularReflection *>(bxdf)->function(__VA_ARGS__); \
    else if (bxdf->type() & BxDF::SpecularTransmission) \
        lvalue op reinterpret_cast<const SpecularTransmission *>(bxdf)->function(__VA_ARGS__); \
    else if (bxdf->type() & BxDF::Phong) \
        lvalue op reinterpret_cast<const Phong *>(bxdf)->function(__VA_ARGS__);

//#define CAST_BXDF_TO_SUBTYPE(pointerVariable, bxdf) \
//    if (bxdf->type() & BxDF::Lambertian) \
//        Lambertian pointerVariable = reinterpret_cast<const Lambertian *>(bxdf)->function(__VA_ARGS__);

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
    RT_FUNCTION BSDF() {  }
    RT_FUNCTION BSDF( const DifferentialGeometry aDiffGeomShading,
                      const optix::float3      & aWorldGeometricNormal )
    {
        _geometricNormal = aWorldGeometricNormal;
        _diffGemetry = aDiffGeomShading;
        _nBxDFs = 0;
        // nvcc give fails without clear error when trying to memset all at once
        memset(_bxdfList, 0, MAX_N_BXDFS * MAX_BXDF_SIZE);
        memset(_bxdfPickProb, 0, MAX_N_BXDFS * sizeof(float));
        _continuationProb = 0.f;
    }

    // For simple case when not differentiating between geometric and shading normal,
    // generates tangent and bitangent
    RT_FUNCTION BSDF( const optix::float3 & aWorldNormal )
    {
        _diffGemetry.SetFromNormal(aWorldNormal);
        _geometricNormal = aWorldNormal;
        _nBxDFs = 0;
        memset(_bxdfList, 0, MAX_N_BXDFS * MAX_BXDF_SIZE);
        memset(_bxdfPickProb, 0, MAX_N_BXDFS * sizeof(float));
        _continuationProb = 0.f;
    }

    RT_FUNCTION unsigned int nBxDFs() const { return _nBxDFs; }

    RT_FUNCTION const DifferentialGeometry & differentialGeometry() const { return _diffGemetry; }

    RT_FUNCTION unsigned int nBxDFs(BxDF::Type aType) const
    {
        unsigned int count = 0;
        for (unsigned int i = 0; i < _nBxDFs; ++i)
        {
            if (bxdfAt(i)->matchFlags(aType))
                ++count;
        }
        return count;
    }

    // Continuation probability for Russian roulette
    RT_FUNCTION float continuationProb() const { return _continuationProb; }

    // Add BxDF. Returns 0 if failed, e.g. MAX_N_BXDFS reached
    RT_FUNCTION int AddBxDF(const BxDF * bxdf)//, uint2 * launchIndex = NULL)
    {
        //OPTIX_PRINTF("AddBxDF - passed BxDF 0x%X\n", sizeof(BxDF * ), (optix::optix_size_t) bxdf);
        if (_nBxDFs == MAX_N_BXDFS) return 0;
        
        // get BxDF list address
        BxDF *pBxDF = reinterpret_cast<BxDF *>(&_bxdfList[_nBxDFs * MAX_BXDF_SIZE]);
        //OPTIX_PRINTF("AddBxDF - this 0x%X BxDF count %d target address 0x%X \n", (optix::optix_size_t) this, _nBxDFs, (optix::optix_size_t) pBxDF);
        memcpy(pBxDF, bxdf, MAX_BXDF_SIZE); // don't care if copy too much, extra bytes won't be used by target type anyway
        
        //Lambertian *lamb = reinterpret_cast<Lambertian*>(const_cast<BxDF*>(bxdf));
        //Lambertian *newLamb = reinterpret_cast<Lambertian*>(const_cast<BxDF*>(pBxDF));
        //OPTIX_PRINTF("AddBxDF - bxdf %8.6f %8.6f %8.6f\n", lamb->_reflectance.x, lamb->_reflectance.y, lamb->_reflectance.z);
        //OPTIX_PRINTF("AddBxDF - set bxdf %8.6f %8.6f %8.6f\n", newLamb->_reflectance.x, newLamb->_reflectance.y, newLamb->_reflectance.z);

        // setting pick probability unweighted by other BxDFs, weighted during sampling based sampled BxDF types
        float pickProb = 0.f;
        CALL_BXDF_CONST_VIRTUAL_FUNCTION(pickProb, +=, pBxDF, reflectProbability);
        CALL_BXDF_CONST_VIRTUAL_FUNCTION(pickProb, +=, pBxDF, transmitProbability); 
        _bxdfPickProb[_nBxDFs] = pickProb;

        // Setting continuation probability explicitly (instead of using arbitrary values) for russian roulette 
        // to make sure the weight of sample never rise
        _continuationProb = optix::fminf(1.f, _continuationProb + pickProb);
        _nBxDFs++;
        //if (launchIndex != NULL)
        //    OPTIX_PRINTFI((*launchIndex), "aBxDF-        _nBxDFs % 14d      _contProb % 14f PickProb[nBxD] % 14f      BxDF type %d \n", _nBxDFs, _continuationProb, _bxdfPickProb[_nBxDFs], pBxDF->type() );
        return 1;
    }

    RT_FUNCTION const BxDF * bxdfAt(const optix::uint & aIndex) const
    {
        return reinterpret_cast<const BxDF *>(&_bxdfList[aIndex * MAX_BXDF_SIZE]);
    }

    RT_FUNCTION const BxDF * bxdfAt(const optix::uint & aIndex, BxDF::Type aType) const
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

    RT_FUNCTION bool isSpecular() const
    {
        return (nBxDFs(BxDF::Type(BxDF::All & ~BxDF::Specular)) == 0);
    }

    // Return bsdf factor for directions oWorldWi and aWorldWi
    // Following typical conventions Wo corresponds to light outgoing direction, 
    // Wi is incident direction. Returns pdf if oPdf not NULL
    RT_FUNCTION optix::float3 f( const optix::float3 & aWorldWo,
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
    //
    // Last parameter aRadianceFromCamera used for VCM to handle specular transmission, since in that
    // case radiance "flows" from camera and light particle weights/importance from light source
    RT_FUNCTION optix::float3 sampleF( const optix::float3 & aWorldWo,
                                       optix::float3       * oWorldWi, 
                                       const optix::float3 & aSample,
                                       float               * oPdfW,
                                       float               * oCosThetaWi = NULL,
                                       BxDF::Type            aSampleType = BxDF::All,
                                       BxDF::Type          * oSampledType = NULL,
                                       const bool            aRadianceFromCamera = false ) const
    {
        // Count matched components.
        unsigned int nMatched = nBxDFs(aSampleType);
        if (nMatched == 0)
        {
            *oPdfW = 0.0f;
            if (oSampledType) *oSampledType = BxDF::Null;
            return optix::make_float3(0.0f);
        }

        float matchedSumContProb = sumContProb(aSampleType);

        //Sample BxDF.
        unsigned int index = optix::min(nMatched-1,
        static_cast<unsigned int>(floorf(aSample.x * static_cast<float>(nMatched))));
        //unsigned int index = 0;
        //unsigned int nMatched = sampleBxDF(aSample.x, aSampleType, &index);

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
        optix::float3 f = optix::make_float3(0.0f);
        optix::float3 wi;
        optix::float2 s = optix::make_float2(aSample.y, aSample.z);
        CALL_BXDF_CONST_VIRTUAL_FUNCTION(f, =, bxdf, sampleF, wo, &wi, s, oPdfW, aRadianceFromCamera);
        
        // Rejected.
        if (*oPdfW == 0.0f)
        {
            if (oSampledType) *oSampledType = BxDF::Null;
            return optix::make_float3(0.0f);
        }

        // Otherwise.
        if (oSampledType) *oSampledType = bxdf->type();
        *oWorldWi = _diffGemetry.ToWorld(wi);
        if (oCosThetaWi) *oCosThetaWi = wi.z;

        // If not specular, sum all non-specular BxDF's probability.
        if (!(bxdf->type() & BxDF::Specular) && nMatched > 1) 
        {
            for (unsigned int i = 0; i < _nBxDFs; i++)
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
    RT_FUNCTION unsigned int sampleBxDF(float sample, BxDF::Type aType, unsigned int * bxdfIndex) const
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


    RT_FUNCTION float sumContProb(BxDF::Type aType) const
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
    bool          _dirFixIsLight;  // true if _localDirFix represents light incident direction
    optix::float3 _localDirFix;    // following convention in SmallVCM, "fix" is corresponds to fixed incident dir stored 
                                   // at hit point opposed to "gen" for generated
public:
    RT_FUNCTION VcmBSDF() : BSDF() { }

    RT_FUNCTION VcmBSDF( const DifferentialGeometry aDiffGeomShading,
                         const optix::float3      & aWorldGeometricNormal,
                         const optix::float3      & aIncidentDir,
                         const bool                 aIncidentDirIsLight ) : 
                BSDF( aDiffGeomShading, aWorldGeometricNormal ) , _dirFixIsLight(aIncidentDirIsLight)
    {
        _localDirFix = _diffGemetry.ToLocal(aIncidentDir);
    }

    // For simple case when not differentiating between geometric and shading normal,
    // generates tangent and bitangent
    RT_FUNCTION VcmBSDF( const optix::float3 & aWorldGeometricNormal,
                         const optix::float3 & aIncidentDir,
                         const bool            aIncidentDirIsLight  ) : BSDF(aWorldGeometricNormal) , _dirFixIsLight(aIncidentDirIsLight)
    {
        _localDirFix = _diffGemetry.ToLocal(aIncidentDir);
    }

    RT_FUNCTION int isValid() const { return EPS_COSINE < _localDirFix.z; }

    RT_FUNCTION bool dirFixIsLight() const { return _dirFixIsLight; }

    RT_FUNCTION optix::float3 localDirFix() const { return _localDirFix; }

    // Return bsdf factor for sampled direction oWorldWi. Returns pdf in and sampled BxDF.
    // Following typical conventions Wo corresponds to light outgoing direction, 
    // Wi is sampled incident direction
    RT_FUNCTION optix::float3 vcmSampleF( optix::float3       * oWorldDirGen,
                                          const optix::float3 & aSample,
                                          float               * oPdfW,
                                          float               * oCosThetaOut, //= NULL,
                                          BxDF::Type            aSampleType = BxDF::All,
                                          BxDF::Type          * oSampledType = NULL ) const
    {
        return sampleF(_diffGemetry.ToWorld(_localDirFix), oWorldDirGen, aSample, 
            oPdfW, oCosThetaOut, aSampleType, oSampledType, _dirFixIsLight);
    }


    // Evaulates pdf for given direction, returns reverse pdf if aEvalRevPdf == true
    // Specular BxDFs are ignored
    RT_FUNCTION float pdf( optix::float3 & oWorldDirGen, bool aEvalRevPdf ) const
    {      
        optix::float3 wo = _diffGemetry.ToLocal(oWorldDirGen);

        float pdf = 0.f;
        for(int i=0; i < _nBxDFs; i++)
        {
            if ( !(bxdfAt(i)->type() & BxDF::Specular) )
                CALL_BXDF_CONST_VIRTUAL_FUNCTION(pdf, +=, bxdfAt(i), pdf, _localDirFix, wo, true);
        }
        return pdf;
    }

#define OPTIX_PRINTFI_ENABLED 0
#define OPTIX_PRINTFID_ENABLED 0

    // Estimates bsdf factor for directions oWorldWi and aWorldWi and pdfs.
    // In typical conventions as when tracing from camera Wo corresponds to light outgoing direction, Wi to 
    // generated incident direction.
    // For VCM evaluation the stored direction localDirFix is used as Wo, generated direction aWorldDirGen as Wi,
    // either when tracing from light or camera. Similarly directPdf corresponds sampling from Wo->Wi, reverse to Wi->Wo
    RT_FUNCTION optix::float3 vcmF( const optix::float3 & aWorldDirGen,
                                    float               & oCosThetaGen,
                                    float               * oDirectPdfW,// = NULL,
                                    float               * oReversePdfW,// = NULL,
                                    const optix::uint2  * dbgLaunchIndex = NULL,
                                    BxDF::Type            aSampleType = BxDF::All ) const
    {
        using namespace optix;

        if (dbgLaunchIndex)
        {
            OPTIX_PRINTFI((*dbgLaunchIndex), "vcmF  -     WorldDirGen % 14f % 14f % 14f \n",
                aWorldDirGen.x, aWorldDirGen.y, aWorldDirGen.z);
            //OPTIX_PRINTFID((*dbgLaunchIndex), "vcmF  - dg  b       %f %f %f\n",
            //    _diffGemetry.bitangent.x, _diffGemetry.bitangent.y, _diffGemetry.bitangent.z);
            //OPTIX_PRINTFID((*dbgLaunchIndex), "vcmF  - dg  t       %f %f %f\n",
            //    _diffGemetry.tangent.x, _diffGemetry.tangent.y, _diffGemetry.tangent.z);
            //OPTIX_PRINTFID((*dbgLaunchIndex), "vcmF  - dg  n       %f %f %f\n",
            //    _diffGemetry.normal.x, _diffGemetry.normal.y, _diffGemetry.normal.z);
        }
        
        float3 localDirGen = _diffGemetry.ToLocal(aWorldDirGen);
        float3 worldDirFix = _diffGemetry.ToWorld(_localDirFix);
        /*OPTIX_PRINTF("vcmF - _localDirFix %f %f %f localDirGen %f %f %f \n",
            _localDirFix.x, _localDirFix.y, _localDirFix.z, localDirGen.x, localDirGen.y, localDirGen.z);*/
        if (dbgLaunchIndex)
        {
            OPTIX_PRINTFI((*dbgLaunchIndex), "vcmF  -    _localDirFix % 14f % 14f % 14f \n", 
                _localDirFix.x, _localDirFix.y, _localDirFix.z);
            OPTIX_PRINTFI((*dbgLaunchIndex), "vcmF  -    _localDirGen % 14f % 14f % 14f \n", 
                localDirGen.x, localDirGen.y, localDirGen.z);
        }

        if (oDirectPdfW) *oDirectPdfW = 0.f;
        if (oReversePdfW) *oReversePdfW = 0.f;

        if (_localDirFix.z < EPS_COSINE || localDirGen.z < EPS_COSINE)
        {
            return make_float3(0.f);
        }

        oCosThetaGen = abs(localDirGen.z);

        // Calculate f.
        if (dot(_geometricNormal, aWorldDirGen) * dot(_geometricNormal, worldDirFix) >= 0.0f)  
            aSampleType = BxDF::Type(aSampleType & ~BxDF::Transmission);      // ignore BTDF
        else
            aSampleType = BxDF::Type(aSampleType & ~BxDF::Reflection);        // ignore BRDF
        
        // Sum all matched BxDF's f and probability
        float3 f = optix::make_float3(0.0f);
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
                if (dbgLaunchIndex)
                {
                    OPTIX_PRINTFI( (*dbgLaunchIndex), "vcmF  -           dPdfW % 14f          rPdfW % 14f \n",
                        dPdfW, rPdfW );
                }
            }
        }

        if (1 < numMatched)
        {
            if (oDirectPdfW) *oDirectPdfW /= static_cast<float>(numMatched);
            if (oReversePdfW) *oReversePdfW /= static_cast<float>(numMatched);
        }
        //OPTIX_PRINTF("vcmF - _nBxDFs %d numMatched %d f %f %f %f \n", _nBxDFs, numMatched, f.x, f.y, f.z);
        if (dbgLaunchIndex)
        {
            OPTIX_PRINTFI((*dbgLaunchIndex), "vcmF  -         _nBxDFs % 14u     numMatched % 14d \n", _nBxDFs, numMatched);            
            OPTIX_PRINTFI((*dbgLaunchIndex), "vcmF  -          bsdf F % 14f % 14f % 14f \n", f.x, f.y, f.z);
        }

        return f;
    }

};


#undef OPTIX_PRINTF_ENABLED
#undef OPTIX_PRINTFI_ENABLED
#undef OPTIX_PRINTFID_ENABLED