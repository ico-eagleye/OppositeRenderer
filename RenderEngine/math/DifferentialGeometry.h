#include <optix_world.h>
#include "optixu_math_namespace.h"

class DifferentialGeometry 
{
    private:
        optix::float3  normal;
        optix::float3  tangent;
        optix::float3  bitangent;
        optix::float3  mX, mY, mZ;

    public:
    __device__ __forceinline__  DifferentialGeometry()
    {
        mX = make_float3(1.f, 0.f, 0.f);
        mY = make_float3(0.f, 1.f, 0.f);
        mZ = make_float3(0.f, 0.f, 1.f);
    };

    // parameters - bitangent, tangent, normal
    __device__ __forceinline__ DifferentialGeometry(
        const optix::float3 x,
        const optix::float3 y,
        const optix::float3 z
    ) :
        mX(x), mY(y), mZ(z) 
    {}

    // sets from tangent t and normal n
    __device__ __forceinline__ DifferentialGeometry(
        const optix::float3 t,
        const optix::float3 n
    ) :
        mY(t), mZ(n) 
    {
        mX = optix::normalize(optix::cross(t,n));
    }

    // sets from normal
    __device__  void SetFromZ(const optix::float3& z)
    {
        optix::float3 mZ = optix::normalize(z);
        optix::float3 tmpX = (std::abs(mZ.x) > 0.99f) ? make_float3(0.f, 1.f, 0.f) : make_float3(1.f, 0.f, 0.f);
        mY = optix::normalize( optix::cross(mZ, tmpX) );
        mX = optix::cross(mY, mZ);
    }

    __device__  __inline__ optix::float3 ToWorld(const optix::float3& a) const
    {
        return mX * a.x + mY * a.y + mZ * a.z;
    }

    __device__  __inline__ optix::float3 ToLocal(const optix::float3& a) const
    {
        return make_float3(optix::dot(a, mX), optix::dot(a, mY), optix::dot(a, mZ));
    }

    __device__ __forceinline__  const optix::float3 Bitangent() const { return mX; }
    __device__ __forceinline__  const optix::float3 Tangent()   const { return mY; }
    __device__ __forceinline__  const optix::float3 Normal()    const { return mZ; }
};