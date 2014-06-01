// TOLICENSE

#pragma once
//#include <optix_world.h>

struct PathVertex
{
	optix::float3 hitPoint;
    optix::float3 throughput;
    float pathDepth;
	float dVCM;
	float dVC;
	float dVM;
	// bsdf data
	optix::uint materialID;
};