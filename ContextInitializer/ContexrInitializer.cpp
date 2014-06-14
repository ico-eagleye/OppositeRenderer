#include <optixu/optixu_math_namespace.h>
#include "ContexrInitializer.h"
#include "renderer/Light.h"
#include "renderer/RayType.h"
#include "renderer/OptixEntryPoint.h"
#include "scene/Cornell.h"

using namespace ContextTest;
class Cornell;

namespace ContextTest
{
  const unsigned int ContextInitializer::OUTPUT_WIDTH = 512;
  const unsigned int ContextInitializer::OUTPUT_HEIGHT = 512;

  const unsigned int ContextInitializer::SUBPATH_LENGHT_ESTIMATE_LAUNCH_WIDTH = 2;
  const unsigned int ContextInitializer::SUBPATH_LENGHT_ESTIMATE_LAUNCH_HEIGHT = 2;

  void ContextInitializer::initialize(optix::Context context, int deviceOrdinal)
  {
    m_context = context;

    // init
    m_context["localIterationNumber"]->setUint(0);
    m_context["sceneRootObject"]->set(m_context->createGroup());
    optix::Buffer outputBuffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, OUTPUT_WIDTH, OUTPUT_HEIGHT );
    m_context["outputBuffer"]->set(outputBuffer);

    // Light sources buffer
    optix::Buffer lightBuffer = context->createBuffer(RT_BUFFER_INPUT);
    lightBuffer->setFormat(RT_FORMAT_USER);
    lightBuffer->setElementSize(sizeof(Light));
    lightBuffer->setSize(1);
    m_context["lights"]->set( lightBuffer );

    m_context->setRayTypeCount(RayType::NUM_RAY_TYPES);
    m_context->setStackSize(1596);
    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(10000000u); 
    m_context->setExceptionEnabled(RTexception::RT_EXCEPTION_ALL , true);

    // Method specific init
    m_context->setEntryPointCount(OptixEntryPointVCM::NUM_PASSES);
    optix::Program generatorProgram = m_context->createProgramFromPTXFile( "test_generator.cu.ptx", "generator" );
    optix::Program exceptionProgram = m_context->createProgramFromPTXFile( "test_generator.cu.ptx", "exception" );
    optix::Program missProgram = m_context->createProgramFromPTXFile( "test_generator.cu.ptx", "miss");
    m_context->setRayGenerationProgram(OptixEntryPointVCM::LIGHT_ESTIMATE_PASS, generatorProgram);
    m_context->setMissProgram(OptixEntryPointVCM::LIGHT_ESTIMATE_PASS, missProgram);
    m_context->setExceptionProgram(OptixEntryPointVCM::LIGHT_ESTIMATE_PASS, exceptionProgram);

    Cornell scene;
    m_context["sceneRootObject"]->set(scene.getSceneRootGroup(m_context));
    m_context->validate();
    m_context->compile();
  }


  void ContextInitializer::launch()
  {
    printf("OptixEntryPoint::VCM_LIGHT_ESTIMATE_PASS launch dim %d x %d\n",
      SUBPATH_LENGHT_ESTIMATE_LAUNCH_WIDTH, SUBPATH_LENGHT_ESTIMATE_LAUNCH_HEIGHT);

    m_context->launch( OptixEntryPointVCM::LIGHT_ESTIMATE_PASS,
      static_cast<unsigned int>(SUBPATH_LENGHT_ESTIMATE_LAUNCH_WIDTH),
      static_cast<unsigned int>(SUBPATH_LENGHT_ESTIMATE_LAUNCH_HEIGHT) );
  }
}