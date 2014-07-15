#pragma once
#include <optix_world.h>
#include "optixu_math_namespace.h"

class DifferentialGeometry 
{
private:
public:
        // basis matrix columns (row of the inverse)
        optix::float3  bitangent;
        optix::float3  tangent;
        optix::float3  normal;

public:
    RT_FUNCTION  DifferentialGeometry()
    {
        bitangent = make_float3(1.f, 0.f, 0.f);
        tangent   = make_float3(0.f, 1.f, 0.f);
        normal    = make_float3(0.f, 0.f, 1.f);
    };

    // parameters - bitangent, tangent, normal
    RT_FUNCTION DifferentialGeometry(
        const optix::float3 b,
        const optix::float3 t,
        const optix::float3 z
    ) :
        bitangent(b), tangent(t), normal(z) 
    {}

    // sets from tangent t and normal n
    RT_FUNCTION DifferentialGeometry(
        const optix::float3 t,
        const optix::float3 n
    ) :
        tangent(t), normal(n) 
    {
        bitangent = optix::normalize(optix::cross(t,n));
    }

    // sets from normal
    RT_FUNCTION void SetFromNormal(const optix::float3& n)
    {
        normal = optix::normalize(n);
        optix::float3 tmpBiTan = (std::abs(normal.x) > 0.99f) ? make_float3(0.f, 1.f, 0.f) : make_float3(1.f, 0.f, 0.f);
        tangent = optix::normalize( optix::cross(normal, tmpBiTan) );
        bitangent = optix::cross(tangent, normal);
    }

    RT_FUNCTION optix::float3 ToWorld(const optix::float3& a) const
    {
        // basis vectors are columns of a matrix multiplied by a
        return bitangent * a.x + 
               tangent   * a.y + 
               normal    * a.z;
    }

    RT_FUNCTION optix::float3 ToLocal(const optix::float3& a) const
    {
        // a multiplied by the inverse of the basis matrix
        return make_float3(optix::dot(bitangent, a), 
                           optix::dot(tangent,   a), 
                           optix::dot(normal,    a));
    }

    RT_FUNCTION  const optix::float3 Bitangent() const { return bitangent; }
    RT_FUNCTION  const optix::float3 Tangent()   const { return tangent; }
    RT_FUNCTION  const optix::float3 Normal()    const { return normal; }
};