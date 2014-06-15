#pragma once

#include <optixu/optixpp_namespace.h>

namespace ContextTest
{
  class ContextInitializer
  {
  public:
    ContextInitializer() {}
    void initialize(optix::Context context, int deviceOrdinal);
    void launch();


  private:
    optix::Context m_context;

    const static unsigned int OUTPUT_WIDTH;
    const static unsigned int OUTPUT_HEIGHT;

    const static unsigned int SUBPATH_LENGHT_ESTIMATE_LAUNCH_WIDTH;
    const static unsigned int SUBPATH_LENGHT_ESTIMATE_LAUNCH_HEIGHT;
  };

}
