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
  const unsigned int ContextInitializer::SUBPATH_LENGHT_ESTIMATE_LAUNCH_WIDTH = 4;
  const unsigned int ContextInitializer::SUBPATH_LENGHT_ESTIMATE_LAUNCH_HEIGHT = 4;

  void ContextInitializer::initializePrograms(optix::Context context, int deviceOrdinal)
  {
    m_context = context;

    // Initialization flow/variables resemble those used in OppositeRenderer to 
    // similar use of context. Some of them are not used in kernels (localIterationNumber,
    // lights etc)

    //// init
    //m_context->setDevices(&deviceOrdinal, &deviceOrdinal+1);
    //m_context["localIterationNumber"]->setUint(0);
    //m_outputBuffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, 1, 1 );
    //m_context["outputBuffer"]->set(m_outputBuffer);

    //// Light sources buffer
    //optix::Buffer lightBuffer = m_context->createBuffer(RT_BUFFER_INPUT);
    //lightBuffer->setFormat(RT_FORMAT_USER);
    //lightBuffer->setElementSize(sizeof(Light));
    //lightBuffer->setSize(1);
    //m_context["lights"]->set( lightBuffer );

    m_context->setRayTypeCount(RayType::NUM_RAY_TYPES);
    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(10000000u); 
    m_context->setExceptionEnabled(RTexception::RT_EXCEPTION_ALL , true);

    //// Method specific init
    //m_lightVertexCountBuffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, 
    //    SUBPATH_LENGHT_ESTIMATE_LAUNCH_WIDTH, SUBPATH_LENGHT_ESTIMATE_LAUNCH_HEIGHT );
    //m_context["lightVertexCountBuffer"]->set(m_lightVertexCountBuffer);

    //m_lightVertexBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, 1u);
    //m_context["lightVertexBuffer"]->set(m_lightVertexBuffer);

    //m_lightVertexBufferIndexBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, 1u);
    //m_context["lightVertexBufferIndexBuffer"]->set(m_lightVertexBufferIndexBuffer);

    //optix::uint* bufferHost = static_cast<optix::uint*>(m_lightVertexBufferIndexBuffer->map());
    //memset(bufferHost, 0, sizeof(optix::uint));
    //m_lightVertexBufferIndexBuffer->unmap();

    m_context->setEntryPointCount(OptixEntryPointVCM::NUM_PASSES);
    optix::Program generatorProgram = m_context->createProgramFromPTXFile( "test_generator.cu.ptx", "generator" );
    optix::Program exceptionProgram = m_context->createProgramFromPTXFile( "test_generator.cu.ptx", "exception" );
    optix::Program missProgram = m_context->createProgramFromPTXFile( "test_generator.cu.ptx", "miss");
    m_context->setRayGenerationProgram(OptixEntryPointVCM::LIGHT_ESTIMATE_PASS, generatorProgram);
    m_context->setExceptionProgram(OptixEntryPointVCM::LIGHT_ESTIMATE_PASS, exceptionProgram);
    m_context->setMissProgram(RayType::LIGHT_VCM, missProgram);

    // Callable program
    optix::Program bsdfEvalProgram = context->createProgramFromPTXFile( "test_generator.cu.ptx", "vcmBsdfEvaluate" );
    m_context["vcmBsdfEvalDiffuse"]->setProgramId(bsdfEvalProgram);

    m_context->validate();
  }


  void ContextInitializer::initializeScene()
  {
    // Initialization flow/variables resemble those used in OppositeRenderer to 
    // similar use of context. Some of them are not used in kernels (localIterationNumber,
    // lights etc)

    //m_context["sceneRootObject"]->set(m_context->createGroup());

    Cornell cornell;
    RootGroup gg = cornell.getSceneRootGroup(m_context); // RootGroup typedef is Group or GeometryGroup, was testing that makes a difference
    m_context["sceneRootObject"]->set(gg);
    m_context->validate();
    m_context->compile();
  }


  // Passing in size since it can change between between launches in the OppositeRenderer where
  // ContextInitializer was used to test the hangs
  void ContextInitializer::launch(unsigned int outputBufWidth, unsigned int outputBufheight)
  {
    //m_outputBuffer->setSize(outputBufWidth, outputBufWidth);

    printf("OptixEntryPoint::LIGHT_ESTIMATE_PASS vertex count estimate launch dim %d x %d\n",
      SUBPATH_LENGHT_ESTIMATE_LAUNCH_WIDTH, SUBPATH_LENGHT_ESTIMATE_LAUNCH_HEIGHT);

    //m_context["lightVertexCountEstimatePass"]->setUint(1u);
    m_context->launch( OptixEntryPointVCM::LIGHT_ESTIMATE_PASS,
        SUBPATH_LENGHT_ESTIMATE_LAUNCH_WIDTH, SUBPATH_LENGHT_ESTIMATE_LAUNCH_HEIGHT );

    //// Get average path length / vertex count
    //optix::uint* buffer_Host = (optix::uint*)m_lightVertexCountBuffer->map();
    //unsigned int subpathCount = SUBPATH_LENGHT_ESTIMATE_LAUNCH_WIDTH * SUBPATH_LENGHT_ESTIMATE_LAUNCH_HEIGHT;
    //unsigned long long sumLen = 0;
    //unsigned long long maxLen = 0;
    //           
    //for(int i = 0; i < subpathCount; i++)
    //{
    //    unsigned long long count = buffer_Host[i];
    //    if (maxLen < count) maxLen = count;
    //    if (0 < count) sumLen += count;
    //}
    //m_lightVertexCountBuffer->unmap();

    //float avgSubpathLength = float(sumLen) / subpathCount;
    //printf("Estimate launch stats: %u  vertices: %u  avgLen: %.4f  maxLen: %u\n\n",
    //    subpathCount, sumLen, avgSubpathLength, maxLen);
  }
}