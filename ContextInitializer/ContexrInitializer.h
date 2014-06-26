#pragma once

#include <optixu/optixpp_namespace.h>

namespace ContextTest
{
  class ContextInitializer
  {
  public:
    ContextInitializer() {}
    void initializePrograms(optix::Context context, int deviceOrdinal);
    void initializeScene();
    void launch(unsigned int outputBufWidth, unsigned int outputBufheight);


  private:
    optix::Context m_context;
    optix::Buffer m_outputBuffer;
    optix::Buffer m_lightVertexCountBuffer;
    optix::Buffer m_lightVertexBuffer;
    optix::Buffer m_lightVertexBufferIndexBuffer;
    const static unsigned int SUBPATH_LENGHT_ESTIMATE_LAUNCH_WIDTH;
    const static unsigned int SUBPATH_LENGHT_ESTIMATE_LAUNCH_HEIGHT;
  };

}
