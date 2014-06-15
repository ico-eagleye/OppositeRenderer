#pragma once

#include <optixu/optixpp_namespace.h>
#include "renderer/Light.h"

#define USE_GEOMETRY_GROUP_AS_ROOT

#ifdef USE_GEOMETRY_GROUP_AS_ROOT
typedef optix::GeometryGroup RootGroup;
#else
typedef optix::Group RootGroup;
#endif

namespace ContextTest
{
  class Cornell
  {
  public:
    Cornell();
    RootGroup getSceneRootGroup(optix::Context & context);

  private:
    optix::GeometryInstance createParallelogram( optix::Context& context,
      const optix::float3& anchor,
      const optix::float3& offset1,
      const optix::float3& offset2,
      const optix::Material& material,
      const optix::float3& color );

    std::vector<Light> m_sceneLights;
    optix::Program m_pgram_bounding_box;
    optix::Program m_pgram_intersection;
  };
}
