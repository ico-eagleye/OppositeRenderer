#include "../ContextInitializer/ContexrInitializer.h"
#include <optixu/optixpp_namespace.h>
#include <iostream>

using namespace ContextTest;

int main( int argc, char** argv )
{
  setbuf(stdout, NULL);
  try 
  {
    optix::Context context = optix::Context::create();
    ContextInitializer ci;
    printf("Initializing...\n");
    ci.initialize(context, 0);
    printf("Launching...\n");
    ci.launch(100u, 100u);
    printf("Launched successfully\n");
  }
  catch (optix::Exception e)
  {
    printf("OptiX exception occured:\n%s", e.getErrorString().c_str());
  }
  catch (std::exception e)
  {
    printf("Exception occured:\n%s", e.what());
  }
  std::cin.ignore();
  return 0;
}