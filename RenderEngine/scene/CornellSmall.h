/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Valdis Vilcans
*/

#ifndef CORNELLSMALL_H
#define CORNELLSMALL_H

#include "Scene.h"
#include "renderer/Light.h"
#include "renderer/Camera.h"
#include "render_engine_export_api.h"




class Material;

class CornellSmall : public IScene
{
public:
    enum Config
    {
        SmallVCMColors     = 1<<0,
        LightArea          = 1<<1,
        LightAreaUpwards   = 1<<2,
        LightPoint         = 1<<3,
        LightPointStrong   = 1<<4,
        LightPointDistant  = 1<<5,
        BackwallBlue       = 1<<6,
        FloorMirror        = 1<<7,
        FloorGlossy        = 1<<8,
        Blocks             = 1<<9,
        LargeMirrorSphere  = 1<<10,
        LargeGlassSphere   = 1<<11,
        SmallMirrorSphere  = 1<<12,
        SmallGlassSphere   = 1<<13,
        Default            = (LightArea | Blocks)
    };

    RENDER_ENGINE_EXPORT_API CornellSmall(void) : m_config(Config::Default) { initialize(); };
    RENDER_ENGINE_EXPORT_API CornellSmall(uint config) : m_config(config) { initialize(); }
    RENDER_ENGINE_EXPORT_API virtual ~CornellSmall(void) {}
    RENDER_ENGINE_EXPORT_API virtual optix::Group getSceneRootGroup(optix::Context & context);
    RENDER_ENGINE_EXPORT_API virtual const QVector<Light> & getSceneLights() const;
    RENDER_ENGINE_EXPORT_API virtual Camera getDefaultCamera(void) const;
    RENDER_ENGINE_EXPORT_API virtual const char* getSceneName() const;
    RENDER_ENGINE_EXPORT_API static const char* getCornellSceneName();
    RENDER_ENGINE_EXPORT_API virtual unsigned int getNumTriangles() const;
    RENDER_ENGINE_EXPORT_API virtual unsigned int getNumMeshes() const;
    RENDER_ENGINE_EXPORT_API virtual AAB getSceneAABB() const;
    //RENDER_ENGINE_EXPORT_API virtual float getSceneInitialPPMRadiusEstimate() const;

private:
    void initialize();

    uint            m_config;
    optix::Material m_material;
    optix::Material m_glassMaterial;
    optix::Program m_pgram_bounding_box;
    optix::Program m_pgram_intersection;
    QVector<Light> m_sceneLights;
    AAB m_sceneAABB;
    optix::GeometryInstance createParallelogram(
        unsigned int meshId,
        optix::Context & context,
        const optix::float3& anchor,
        const optix::float3& offset1,
        const optix::float3& offset2,
        Material & material
    );
};
#endif