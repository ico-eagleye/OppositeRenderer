/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include "CornellSmall.h"
#include "renderer/RayType.h"
#include <optixu/optixpp_namespace.h>
#include "geometry_instance/SphereInstance.h"
#include "geometry_instance/Transform.h"
#include "geometry_instance/AABInstance.h"
#include "math/Sphere.h"
#include "material/Glass.h"
#include "material/Diffuse.h"
#include "material/ParticipatingMedium.h"
#include "material/Mirror.h"
#include "material/DiffuseEmitter.h"

CornellSmall::CornellSmall(void)
{
    optix::float3 anchor = optix::make_float3( 1.f, 2.499f, 1.f);
    optix::float3 v1 = optix::make_float3( 0.5f, 0.0f, 0.0f);
    optix::float3 v2 = optix::make_float3( 0.0f, 0.0f, 0.5f);
    //optix::float3 power = optix::make_float3( 0.5e6f, 0.4e6f, 0.2e6f );
    //optix::float3 power = optix::make_float3( 0.75f * 5.f );
    //optix::float3 power = optix::make_float3( M_PIf );
    optix::float3 power = optix::make_float3( 19.661107023935260172519494336416f );
    Light light(power, anchor, v1, v2);
    m_sceneLights.push_back(light);

    m_sceneAABB.min = Vector3(-0.1f);
    m_sceneAABB.max = Vector3(2.5f, 2.5f, 2.5f) + 0.1f;
}

optix::GeometryInstance CornellSmall::createParallelogram(
    unsigned int meshId,
    optix::Context & context,
    const optix::float3& anchor,
    const optix::float3& offset1,
    const optix::float3& offset2,
    Material & material)
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

    parallelogram["meshId"]->setUint(meshId);
    parallelogram["plane"]->setFloat( plane );
    parallelogram["anchor"]->setFloat( anchor );
    parallelogram["v1"]->setFloat( v1 );
    parallelogram["v2"]->setFloat( v2 );

    optix::Material matl = material.getOptixMaterial(context);

    optix::GeometryInstance gi = context->createGeometryInstance( parallelogram, &matl, &matl+1 );
    material.registerGeometryInstanceValues(gi);
    return gi;
}


optix::Group CornellSmall::getSceneRootGroup(optix::Context & context)
{
    using namespace optix;

    m_pgram_bounding_box = context->createProgramFromPTXFile( "parallelogram.cu.ptx", "bounds" );
    m_pgram_intersection = context->createProgramFromPTXFile( "parallelogram.cu.ptx", "intersect" );

    // create geometry instances
    QVector<optix::GeometryInstance> gis;

    Diffuse diffuseWhite = Diffuse(optix::make_float3( 0.8f ));
    Diffuse diffuseGreen = Diffuse(optix::make_float3( 0.05f, 0.8f, 0.05f ));
    Diffuse diffuseRed = Diffuse(optix::make_float3( 1.f, 0.05f, 0.05f ));

    // Cornell box size in SmallVCM 2.56004
    // Floor
    gis.push_back( createParallelogram(0, context, optix::make_float3( 0.0f, 0.0f, 0.0f ),
        optix::make_float3( 0.0f, 0.0f, 2.5f ),
        optix::make_float3( 2.5f, 0.0f, 0.0f ),
        diffuseWhite ) );

    // Ceiling
    gis.push_back( createParallelogram(1, context, optix::make_float3( 0.0f, 2.5f, 0.0f ),
        optix::make_float3( 2.5f, 0.0f, 0.0f ),
        optix::make_float3( 0.0f, 0.0f, 2.5f ),
        diffuseWhite ) );

    // Back wall
    gis.push_back( createParallelogram(2, context,optix::make_float3( 0.0f, 0.0f, 2.5f),
        optix::make_float3( 0.0f, 2.5f, 0.0f),
        optix::make_float3( 2.5f, 0.0f, 0.0f),
        diffuseWhite));

    // Right wall
    gis.push_back( createParallelogram(3, context, optix::make_float3( 0.0f, 0.0f, 0.0f ),
        optix::make_float3( 0.0f, 2.5f, 0.0f ),
        optix::make_float3( 0.0f, 0.0f, 2.5f ),
        diffuseGreen ) );

    // Left wall
    gis.push_back( createParallelogram(4, context, optix::make_float3( 2.5f, 0.0f, 0.0f ),
        optix::make_float3( 0.0f, 0.0f, 2.5f ),
        optix::make_float3( 0.0f, 2.5f, 0.0f ),
        diffuseRed ) );

    //// Short block
    //gis.push_back( createParallelogram(5, context, 
    //    optix::make_float3( 130.0f, 165.0f, 65.0f) / 220.f,
    //    optix::make_float3( -48.0f, 0.0f, 160.0f) / 220.f,
    //    optix::make_float3( 160.0f, 0.0f, 49.0f) / 220.f,
    //    diffuseWhite ) );
    //gis.push_back( createParallelogram(6, context, 
    //    optix::make_float3( 290.0f, 0.0f, 114.0f) / 220.f,
    //    optix::make_float3( 0.0f, 165.0f, 0.0f) / 220.f,
    //    optix::make_float3( -50.0f, 0.0f, 158.0f) / 220.f,
    //    diffuseWhite ) );
    //gis.push_back( createParallelogram(7, context, 
    //    optix::make_float3( 130.0f, 0.0f, 65.0f) / 220.f,
    //    optix::make_float3( 0.0f, 165.0f, 0.0f) / 220.f,
    //    optix::make_float3( 160.0f, 0.0f, 49.0f) / 220.f,
    //    diffuseWhite ) );
    //gis.push_back( createParallelogram(8, context, 
    //    optix::make_float3( 82.0f, 0.0f, 225.0f) / 220.f,
    //    optix::make_float3( 0.0f, 165.0f, 0.0f) / 220.f,
    //    optix::make_float3( 48.0f, 0.0f, -160.0f) / 220.f,
    //    diffuseWhite ) );
    //gis.push_back( createParallelogram(9, context,
    //    optix::make_float3( 240.0f, 0.0f, 272.0f) / 220.f,
    //    optix::make_float3( 0.0f, 165.0f, 0.0f) / 220.f,
    //    optix::make_float3( -158.0f, 0.0f, -47.0f) / 220.f,
    //    diffuseWhite));
    //    
    //// Tall block
    //gis.push_back( createParallelogram(10, context, 
    //    optix::make_float3( 423.0f, 340.0f, 247.0f) / 220.f,
    //    optix::make_float3( -158.0f, 0.0f, 49.0f) / 220.f,
    //    optix::make_float3( 49.0f, 0.0f, 159.0f) / 220.f,
    //    diffuseWhite ) );
    //gis.push_back( createParallelogram(11, context, 
    //    optix::make_float3( 423.0f, 0.0f, 247.0f) / 220.f,
    //    optix::make_float3( 0.0f, 340.0f, 0.0f) / 220.f,
    //    optix::make_float3( 49.0f, 0.0f, 159.0f) / 220.f,
    //    diffuseWhite ) );
    //gis.push_back( createParallelogram(12, context, 
    //    optix::make_float3( 472.0f, 0.0f, 406.0f) / 220.f,
    //    optix::make_float3( 0.0f, 340.0f, 0.0f) / 220.f,
    //    optix::make_float3( -158.0f, 0.0f, 50.0f) / 220.f,
    //    diffuseWhite ) );
    //gis.push_back( createParallelogram(13, context, 
    //    optix::make_float3( 314.0f, 0.0f, 456.0f) / 220.f,
    //    optix::make_float3( 0.0f, 340.0f, 0.0f) / 220.f,
    //    optix::make_float3( -49.0f, 0.0f, -160.0f) / 220.f,
    //    diffuseWhite ) );
    //gis.push_back( createParallelogram(14, context, 
    //    optix::make_float3( 265.0f, 0.0f, 296.0f) / 220.f,
    //    optix::make_float3( 0.0f, 340.1f, 0.0f) / 220.f,
    //    optix::make_float3( 158.0f, 0.0f, -49.0f) / 220.f,
    //    diffuseWhite ) );
        
    // Light

    DiffuseEmitter emitter = DiffuseEmitter(m_sceneLights[0].power, Vector3(1));
    emitter.setInverseArea(m_sceneLights[0].inverseArea);
    for(int i = 0; i < m_sceneLights.size(); i++)
    {
        gis.push_back(createParallelogram(15 + i, context, m_sceneLights[i].position, m_sceneLights[i].v1, m_sceneLights[i].v2, emitter));
    }

    Glass glass = Glass(1.5, optix::make_float3(1.f,1.f,1.f));
    Mirror mirror = Mirror(optix::make_float3(0.7f,0.7f,1.f));
    Diffuse diff(optix::make_float3(0.f,1.f,0.f));

    // Participating Tests

    /*#if ENABLE_PARTICIPATING_MEDIA
    ParticipatingMedium partmedium = ParticipatingMedium(0.001, 0.00);
    AABInstance participatingMediumCube (partmedium, AAB(Vector3(-1), Vector3(556.0f, 548.85f, 559.2f)-1));
    gis.push_back(participatingMediumCube.getOptixGeometryInstance(context));
    SphereInstance sphere = SphereInstance(glass, Sphere(Vector3(250, 370, 250), 50));
    gis.push_back(sphere.getOptixGeometryInstance(context));
#endif
    SphereInstance sphere2 = SphereInstance(glass, Sphere(Vector3(450, 50, 300), 50));
    gis.push_back(sphere2.getOptixGeometryInstance(context));*/

    // Create geometry group
    optix::GeometryGroup geometry_group = context->createGeometryGroup();
    geometry_group->setChildCount( static_cast<unsigned int>( gis.size() ) );
    for (int i = 0; i < gis.size(); ++i )
        geometry_group->setChild( i, gis[i] );

    // vmarz: Changed to Trbvh. Sbvh can cause weird hangs 
    // https://devtalk.nvidia.com/default/topic/751906/optix/weird-ray-generation-hang-really-simple-code-/
    geometry_group->setAcceleration(context->createAcceleration("NoAccel", "NoAccel")); // Bvh Sbvh Trbvh NoAccel

    optix::Group gro = context->createGroup();
    gro->setChildCount(1);
    gro->setChild(0, geometry_group);
    optix::Acceleration acceleration = context->createAcceleration("NoAccel", "NoAccel"); // Bvh BvhCompact NoAccel
    gro->setAcceleration(acceleration);

    return gro;
}

const QVector<Light> & CornellSmall::getSceneLights(void) const
{
    return m_sceneLights;
}

Camera CornellSmall::getDefaultCamera(void) const
{
    return Camera(
        optix::make_float3( 1.25f, 1.25f, -2.85f ),
        optix::make_float3( 1.25f, 1.25f, 0.0f ),
        optix::make_float3( 0.0f, 1.0f,  0.0f ),
        45.0f, 
        45.0f,
        Camera::KeepVertical );
}

const char* CornellSmall::getSceneName() const
{
    return CornellSmall::getCornellSceneName();
}

AAB CornellSmall::getSceneAABB() const
{
    return m_sceneAABB;
}

const char* CornellSmall::getCornellSceneName()
{
     return "CornellSmall";
}

unsigned int CornellSmall::getNumTriangles() const
{
    return 16; // should be automatically calculated from the getSceneRootGroup count
}

unsigned int CornellSmall::getNumMeshes() const
{
    return 16; // should be automatically calculated from the getSceneRootGroup count
}

