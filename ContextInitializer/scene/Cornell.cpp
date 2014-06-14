#pragma once

#include "scene/Cornell.h"
#include "renderer/RayType.h"
#include <optixu/optixu_math_namespace.h>

namespace ContextTest
{

  Cornell::Cornell()
  {
    optix::float3 anchor = optix::make_float3( 343.0f, 548.7999f, 227.0f);
    optix::float3 v1 = optix::make_float3( 0.0f, 0.0f, 105.0f);
    optix::float3 v2 = optix::make_float3( -130.0f, 0.0f, 0.0f);
    optix::float3 power = optix::make_float3( 0.5e6f, 0.4e6f, 0.2e6f );
    Light light(power, anchor, v1, v2);
    m_sceneLights.push_back(light);
  }

  optix::GeometryInstance Cornell::createParallelogram( 
    optix::Context& context,
    const optix::float3& anchor,
    const optix::float3& offset1,
    const optix::float3& offset2,
    const optix::Material& material,
    const optix::float3& color )
  {
    optix::Geometry parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount( 1u );
    parallelogram->setIntersectionProgram( m_pgram_intersection );
    parallelogram->setBoundingBoxProgram( m_pgram_bounding_box );

    optix::float3 normal = optix::normalize( optix::cross( offset1, offset2 ) );
    float d = optix::dot( normal, anchor );
    optix::float4 plane = optix::make_float4( normal, d );

    optix::float3 v1 = offset1 / optix::dot( offset1, offset1 );
    optix::float3 v2 = offset2 / optix::dot( offset2, offset2 );

    parallelogram["plane"]->setFloat( plane );
    parallelogram["anchor"]->setFloat( anchor );
    parallelogram["v1"]->setFloat( v1 );
    parallelogram["v2"]->setFloat( v2 );

    optix::GeometryInstance gi = context->createGeometryInstance(parallelogram, &material, &material+1);
    gi["Kd"]->setFloat(color);
    return gi;
  }


  optix::Group Cornell::getSceneRootGroup(optix::Context & context)
  {
    m_pgram_bounding_box = context->createProgramFromPTXFile( "test_parallelogram.cu.ptx", "bounds" );
    m_pgram_intersection = context->createProgramFromPTXFile( "test_parallelogram.cu.ptx", "intersect" );

    optix::float3 diffuseWhite = optix::make_float3( 0.8f );
    optix::float3 diffuseGreen = optix::make_float3( 0.05f, 0.8f, 0.05f );
    optix::float3 diffuseRed = optix::make_float3( 1.f, 0.05f, 0.05f );

    optix::Material material = context->createMaterial();
    optix::Program vcmLightProgram = context->createProgramFromPTXFile( "test_hit.cu.ptx", "closestHit");
    material->setClosestHitProgram(RayType::LIGHT_VCM, vcmLightProgram);
    material->validate();

    std::vector<optix::GeometryInstance> gis;

    // Floor
    gis.push_back( createParallelogram(context, optix::make_float3( 0.0f, 0.0f, 0.0f ),
      optix::make_float3( 0.0f, 0.0f, 559.2f ),
      optix::make_float3( 556.0f, 0.0f, 0.0f ),
      material, diffuseWhite ) );

    // Ceiling
    gis.push_back( createParallelogram(context, optix::make_float3( 0.0f, 548.80f, 0.0f ),
      optix::make_float3( 556.0f, 0.0f, 0.0f ),
      optix::make_float3( 0.0f, 0.0f, 559.2f ),
      material, diffuseWhite));

    // Back wall
    gis.push_back( createParallelogram(context,optix::make_float3( 0.0f, 0.0f, 559.2f),
      optix::make_float3( 0.0f, 548.8f, 0.0f),
      optix::make_float3( 556.0f, 0.0f, 0.0f),
      material, diffuseWhite));

    // Right wall
    gis.push_back( createParallelogram(context, optix::make_float3( 0.0f, 0.0f, 0.0f ),
      optix::make_float3( 0.0f, 548.8f, 0.0f ),
      optix::make_float3( 0.0f, 0.0f, 559.2f ),
      material, diffuseGreen));

    // Left wall
    gis.push_back( createParallelogram(context, optix::make_float3( 556.0f, 0.0f, 0.0f ),
      optix::make_float3( 0.0f, 0.0f, 559.2f ),
      optix::make_float3( 0.0f, 548.8f, 0.0f ),
      material, diffuseRed) );

    // Short block
    gis.push_back( createParallelogram(context, optix::make_float3( 130.0f, 165.0f, 65.0f),
      optix::make_float3( -48.0f, 0.0f, 160.0f),
      optix::make_float3( 160.0f, 0.0f, 49.0f),
      material, diffuseWhite) );
    gis.push_back( createParallelogram(context, optix::make_float3( 290.0f, 0.0f, 114.0f),
      optix::make_float3( 0.0f, 165.0f, 0.0f),
      optix::make_float3( -50.0f, 0.0f, 158.0f),
      material, diffuseWhite) );
    gis.push_back( createParallelogram(context, optix::make_float3( 130.0f, 0.0f, 65.0f),
      optix::make_float3( 0.0f, 165.0f, 0.0f),
      optix::make_float3( 160.0f, 0.0f, 49.0f),
      material, diffuseWhite) );
    gis.push_back( createParallelogram(context, optix::make_float3( 82.0f, 0.0f, 225.0f),
      optix::make_float3( 0.0f, 165.0f, 0.0f),
      optix::make_float3( 48.0f, 0.0f, -160.0f),
      material, diffuseWhite) );
    gis.push_back( createParallelogram(context, optix::make_float3( 240.0f, 0.0f, 272.0f),
      optix::make_float3( 0.0f, 165.0f, 0.0f),
      optix::make_float3( -158.0f, 0.0f, -47.0f),
      material, diffuseWhite) );

    // Tall block
    gis.push_back( createParallelogram(context, optix::make_float3( 423.0f, 340.0f, 247.0f),
      optix::make_float3( -158.0f, 0.0f, 49.0f),
      optix::make_float3( 49.0f, 0.0f, 159.0f),
      material, diffuseWhite) );
    gis.push_back( createParallelogram(context, optix::make_float3( 423.0f, 0.0f, 247.0f),
      optix::make_float3( 0.0f, 340.0f, 0.0f),
      optix::make_float3( 49.0f, 0.0f, 159.0f),
      material, diffuseWhite) );
    gis.push_back( createParallelogram(context, optix::make_float3( 472.0f, 0.0f, 406.0f),
      optix::make_float3( 0.0f, 340.0f, 0.0f),
      optix::make_float3( -158.0f, 0.0f, 50.0f),
      material, diffuseWhite) );
    gis.push_back( createParallelogram(context, optix::make_float3( 314.0f, 0.0f, 456.0f),
      optix::make_float3( 0.0f, 340.0f, 0.0f),
      optix::make_float3( -49.0f, 0.0f, -160.0f),
      material, diffuseWhite) );
    gis.push_back( createParallelogram(context, optix::make_float3( 265.0f, 0.0f, 296.0f),
      optix::make_float3( 0.0f, 340.1f, 0.0f),
      optix::make_float3( 158.0f, 0.0f, -49.0f),
      material, diffuseWhite) );

    // skiped adding light

    optix::GeometryGroup geometry_group = context->createGeometryGroup();
    geometry_group->setChildCount( static_cast<unsigned int>( gis.size() ) );
    for (int i = 0; i < gis.size(); ++i )
      geometry_group->setChild( i, gis[i] );

    geometry_group->setAcceleration(context->createAcceleration("Trbvh", "Bvh"));

    optix::Group gro = context->createGroup();
    gro->setChildCount(1);
    gro->setChild(0, geometry_group);
    optix::Acceleration acceleration = context->createAcceleration("Trbvh", "Bvh");
    gro->setAcceleration(acceleration);
    return gro;
  }
}
