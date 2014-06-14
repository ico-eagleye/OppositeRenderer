#include "../ContextInitializer/ContexrInitializer.h"
#include <optixu/optixpp_namespace.h>

using namespace ContextTest;

int main( int argc, char** argv )
{
  optix::Context context = optix::Context::create();
  ContextInitializer ci;
  ci.initialize(context, 0);
  ci.launch();
  return 0;
}