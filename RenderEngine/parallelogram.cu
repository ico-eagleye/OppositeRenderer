
/*
* Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and proprietary
* rights in and to this software, related documentation and any modifications thereto.
* Any use, reproduction, disclosure or distribution of this software and related
* documentation without an express license agreement from NVIDIA Corporation is strictly
* prohibited.
*
* TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
* INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
* SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
* LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
* BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
* INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
* SUCH DAMAGES
*/


#include <optix_world.h>
#include <optix_cuda.h>
#include "config.h"

using namespace optix;

// for mesh intesection count
rtDeclareVariable(uint, meshId, , );
rtBuffer<uint, 1> hitsPerMeshBuffer;

// parallelogram 
rtDeclareVariable(float4, plane, , ); // vmarz: xyz=normal, w=dot(normal, anchor)=D dist to plane form origin
									  // in plane definition in Hesse normal form
rtDeclareVariable(float3, v1, , );
rtDeclareVariable(float3, v2, , );
rtDeclareVariable(float3, anchor, , );
rtDeclareVariable(int, lgt_instance, , ) = {0};

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, ); 
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, ); 
rtDeclareVariable(int, lgt_idx, attribute lgt_idx, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void intersect(int primIdx)
{
	float3 n = make_float3( plane );
	float dt = dot(ray.direction, n );				// vmarz: cos between ray dir and n
	float t = (plane.w - dot(n, ray.origin))/dt;	// (dist to plane - len origin proj on n) / scaled by cos [e.g. t grows angle grows]
	if( t > ray.tmin && t < ray.tmax ) {
		float3 p = ray.origin + ray.direction * t;
		float3 vi = p - anchor;
		float a1 = dot(v1, vi);
		if(a1 >= 0 && a1 <= 1)					// vmarz: comparing with 1 because v1 was scaled by 1./length^2,
		{										// so any dot products with it cannot be bigger than original v1 length
			float a2 = dot(v2, vi);
			if(a2 >= 0 && a2 <= 1)
			{
				if( rtPotentialIntersection( t ) ) 
				{
					shadingNormal = geometricNormal = n;
					texcoord = make_float3(a1,a2,0);
					lgt_idx = lgt_instance;
#if ENABLE_MESH_HITS_COUNTING
					atomicAdd(&hitsPerMeshBuffer[meshId], 1);
#endif
					rtReportIntersection( 0 );
				}
			}
		}
	}
}

RT_PROGRAM void bounds (int, float result[6])
{
    // v1 and v2 are scaled by 1./length^2.  Rescale back to normal for the bounds computation.
    // vmarz: NOTE not scaled by lenght, so v1, v2 are not unit vectors
	const float3 tv1  = v1 / dot( v1, v1 );
	const float3 tv2  = v2 / dot( v2, v2 );
	const float3 p00  = anchor;
	const float3 p01  = anchor + tv1;
	const float3 p10  = anchor + tv2;
	const float3 p11  = anchor + tv1 + tv2;
	const float  area = length(cross(tv1, tv2));

	optix::Aabb* aabb = (optix::Aabb*)result;

	if(area > 0.0f && !isinf(area)) {
		aabb->m_min = fminf( fminf( p00, p01 ), fminf( p10, p11 ) );
		aabb->m_max = fmaxf( fmaxf( p00, p01 ), fmaxf( p10, p11 ) );
	} else {
		aabb->invalidate();
	}
}

