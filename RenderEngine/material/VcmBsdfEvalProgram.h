#pragma once

#include "renderer/vcm/LightVertex.h"


//__device__ float vcmBsdfEvaluateDiffuse(LightVertex lightVertex, float3 &aDirGen, float3 &oCosThetaGen,
//                      float *oDirectPdfW = NULL, float *oReversePdfW = NULL) 
//{
//    
//    return 42.f;
//}
//
//
//__inline__ __device__ float vcmBsdfEvaluate(LightVertex lightVertex, float3 &aDirGen, float3 &oCosThetaGen,
//                                            float *oDirectPdfW = NULL, float *oReversePdfW = NULL) 
//{
//    switch (lightVertex.bsdfData.material)
//    {
//    case VcmMeterial::DIFFUSE:
//        return vcmBsdfEvaluateDiffuse(lightVertex, aDirGen, oCosThetaGen, oDirectPdfW, oReversePdfW);
//    default:
//        return 0.f;
//    }
//}