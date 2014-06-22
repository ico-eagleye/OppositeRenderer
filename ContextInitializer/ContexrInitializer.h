#pragma once

#include <optixu/optixpp_namespace.h>

namespace ContextTest
{
  class ContextInitializer
  {
  public:
    ContextInitializer() {}
    void initialize(optix::Context context, int deviceOrdinal);
    void launch(unsigned int outputBufWidth, unsigned int outputBufheight);


  private:
    optix::Context m_context;
    optix::Buffer m_outputBuffer;
    optix::Buffer m_lightVertexCountBuffer;
    optix::Buffer m_dbgNoMissHitStops;
    const static unsigned int SUBPATH_LENGHT_ESTIMATE_LAUNCH_WIDTH;
    const static unsigned int SUBPATH_LENGHT_ESTIMATE_LAUNCH_HEIGHT;
  };

}
