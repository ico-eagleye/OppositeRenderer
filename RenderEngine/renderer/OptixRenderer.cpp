/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Stian Pedersen
 *                Valdis Vilcans
*/

#include <cuda.h>
#include <curand_kernel.h>
#include "OptixRenderer.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>
#include "config.h"
#include "RandomState.h"
#include "renderer/OptixEntryPoint.h"
#include "renderer/Hitpoint.h"
#include "renderer/ppm/Photon.h"
#include "Camera.h"
#include <QThread>
#include "renderer/RayType.h"
#include "ComputeDevice.h"
#include "clientserver/RenderServerRenderRequest.h"
#include <exception>
#include "util/sutil.h"
#include "scene/IScene.h"
#include "renderer/helpers/nsight.h"
#include "renderer/helpers/samplers.h"
#include "renderer/vcm/LightVertex.h"
#include "renderer/vcm/config_vcm.h"
#include "renderer/vcm/vcm_shared.h"
#include "util/logging.h"

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_UNIFORM_GRID
const unsigned int OptixRenderer::PHOTON_GRID_MAX_SIZE = 100*100*100;
#else
const unsigned int OptixRenderer::PHOTON_GRID_MAX_SIZE = 0;
#endif

const unsigned int OptixRenderer::MAX_PHOTON_COUNT = MAX_PHOTONS_DEPOSITS_PER_EMITTED;
const unsigned int OptixRenderer::PHOTON_LAUNCH_WIDTH = 1024;
const unsigned int OptixRenderer::PHOTON_LAUNCH_HEIGHT = 1024;
// Ensure that NUM PHOTONS are a power of 2 for stochastic hash

const unsigned int OptixRenderer::EMITTED_PHOTONS_PER_ITERATION = OptixRenderer::PHOTON_LAUNCH_WIDTH*OptixRenderer::PHOTON_LAUNCH_HEIGHT;
const unsigned int OptixRenderer::NUM_PHOTONS = OptixRenderer::EMITTED_PHOTONS_PER_ITERATION*OptixRenderer::MAX_PHOTON_COUNT;

// VCM
const unsigned int OptixRenderer::VCM_MAX_PATH_LENGTH = 10u;
const unsigned int OptixRenderer::VCM_SUBPATH_LEN_ESTIMATE_LAUNCH_WIDTH = 128u;
const unsigned int OptixRenderer::VCM_SUBPATH_LEN_ESTIMATE_LAUNCH_HEIGHT = 128u;

// This would be used for uniform vertex sampling
const unsigned int OptixRenderer::VCM_LIGHT_PASS_LAUNCH_WIDTH = 512u;
const unsigned int OptixRenderer::VCM_LIGHT_PASS_LAUNCH_HEIGHT = 512u;
//const unsigned int OptixRenderer::VCM_NUM_LIGHT_PATH_CONNECTIONS = 3u;
const unsigned int OptixRenderer::VCM_NUM_LIGHT_PATH_CONNECTIONS = 1u;

using namespace optix;

inline unsigned int pow2roundup(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

inline float max(float a, float b)
{
  return a > b ? a : b;
}

OptixRenderer::OptixRenderer() : 
    m_initialized(false),
    m_lightVertexCountEstimated(false),
    m_width(10),
    m_height(10)
{
    try
    {
        m_context = optix::Context::create();
        if(!m_context)
        {
            throw std::exception("Unable to create OptiX context.");
        }
    }
    catch(const optix::Exception & e)
    {
        throw std::exception(e.getErrorString().c_str());
    }
    catch(const std::exception & e)
    {
        QString error = QString("Error during initialization of Optix: %1").arg(e.what());
        throw std::exception(error.toLatin1().constData());
    }
}

OptixRenderer::~OptixRenderer()
{
    printf("Context Destroy\n");
    m_context->destroy();
    cudaDeviceReset();
}

void OptixRenderer::initialize(const ComputeDevice & device)
{
    if(m_initialized)
    {
        throw std::exception("ERROR: Multiple OptixRenderer::initialize!\n");
    }

    initDevice(device);

    m_context->setRayTypeCount(RayType::NUM_RAY_TYPES);
    m_context->setEntryPointCount(OptixEntryPoint::NUM_PASSES);    
    m_context->setStackSize(ENABLE_PARTICIPATING_MEDIA ? 3000 : 1596);

    m_context["maxPhotonDepositsPerEmitted"]->setUint(MAX_PHOTON_COUNT);
    m_context["ppmAlpha"]->setFloat(0);
    m_context["totalEmitted"]->setFloat(0.0f);
    m_context["iterationNumber"]->setFloat(0.0f);
    m_context["localIterationNumber"]->setUint(0);
    m_context["ppmRadius"]->setFloat(0.f);
    m_context["ppmRadiusSquared"]->setFloat(0.f);
    m_context["ppmRadiusSquaredNew"]->setFloat(0.f);
    m_context["ppmDefaultRadius2"]->setFloat(0.f);
    m_context["emittedPhotonsPerIteration"]->setUint(EMITTED_PHOTONS_PER_ITERATION);
    m_context["emittedPhotonsPerIterationFloat"]->setFloat(float(EMITTED_PHOTONS_PER_ITERATION));
    m_context["photonLaunchWidth"]->setUint(PHOTON_LAUNCH_WIDTH);
    m_context["participatingMedium"]->setUint(0);

    // An empty scene root node
    optix::Group group = m_context->createGroup();
    m_context["sceneRootObject"]->set(group);
    //optix::Acceleration acceleration = m_context->createAcceleration("Bvh", "Bvh");
    //group->setAcceleration(acceleration);
    
    // Ray Trace OptixEntryPoint Output Buffer
    m_raytracePassOutputBuffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT );
    m_raytracePassOutputBuffer->setFormat( RT_FORMAT_USER );
    m_raytracePassOutputBuffer->setElementSize( sizeof( Hitpoint ) );
    m_raytracePassOutputBuffer->setSize( m_width, m_height );
    m_context["raytracePassOutputBuffer"]->set( m_raytracePassOutputBuffer );

    // Path Tracing ray generation OptixEntryPoint
    {
        Program generatorProgram = m_context->createProgramFromPTXFile( "RayGeneratorPT.cu.ptx", "generateRay" );
        Program exceptionProgram =  m_context->createProgramFromPTXFile( "RayGeneratorPT.cu.ptx", "exception" );
        Program missProgram = m_context->createProgramFromPTXFile( "RayGeneratorPT.cu.ptx", "miss" );
        m_context->setRayGenerationProgram( OptixEntryPoint::PT_RAYTRACE_PASS, generatorProgram );
        m_context->setExceptionProgram(OptixEntryPoint::PT_RAYTRACE_PASS, exceptionProgram);
    }

    // PPM Ray Generation Program OptixEntryPoint
    {
        Program generatorProgram = m_context->createProgramFromPTXFile( "RayGeneratorPPM.cu.ptx", "generateRay" );
        Program exceptionProgram = m_context->createProgramFromPTXFile( "RayGeneratorPPM.cu.ptx", "exception" );
        Program missProgram = m_context->createProgramFromPTXFile( "RayGeneratorPPM.cu.ptx", "miss" );
        
        m_context->setRayGenerationProgram( OptixEntryPoint::PPM_RAYTRACE_PASS, generatorProgram );
        m_context->setExceptionProgram( OptixEntryPoint::PPM_RAYTRACE_PASS, exceptionProgram );
        m_context->setMissProgram(RayType::RADIANCE, missProgram);
        m_context->setMissProgram(RayType::RADIANCE_IN_PARTICIPATING_MEDIUM, missProgram);
    }

    // PPM Photon Tracing OptixEntryPoint
    {
        Program generatorProgram = m_context->createProgramFromPTXFile( "PhotonGenerator.cu.ptx", "generator" );
        Program exceptionProgram = m_context->createProgramFromPTXFile( "PhotonGenerator.cu.ptx", "exception" );
        Program missProgram = m_context->createProgramFromPTXFile( "PhotonGenerator.cu.ptx", "miss");
        m_context->setRayGenerationProgram(OptixEntryPoint::PPM_PHOTON_PASS, generatorProgram);
        m_context->setMissProgram(OptixEntryPoint::PPM_PHOTON_PASS, missProgram);
        m_context->setExceptionProgram(OptixEntryPoint::PPM_PHOTON_PASS, exceptionProgram);
    }

    m_photons = m_context->createBuffer(RT_BUFFER_OUTPUT);
    m_photons->setFormat( RT_FORMAT_USER );
    m_photons->setElementSize( sizeof( Photon ) );
    m_photons->setSize( NUM_PHOTONS );
    m_context["photons"]->set( m_photons );
    m_context["photonsSize"]->setUint( NUM_PHOTONS );

#pragma region Acceleration structure
#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_STOCHASTIC_HASH

    optix::Buffer photonsHashTableCount = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, NUM_PHOTONS);
    m_context["photonsHashTableCount"]->set(photonsHashTableCount);
    {
        Program program = m_context->createProgramFromPTXFile( "UniformGridPhotonInitialize.cu.ptx", "kernel" );
        m_context->setRayGenerationProgram(OptixEntryPoint::PPM_CLEAR_PHOTONS_UNIFORM_GRID_PASS, program );
    }
    m_context["photonsGridCellSize"]->setFloat(0.0f);
    m_context["photonsGridCellSize"]->setFloat(0.0f);
    m_context["photonsGridSize"]->setUint(0,0,0);
    m_context["photonsWorldOrigo"]->setFloat(make_float3(0));

#elif ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_KD_TREE_CPU

    m_photonKdTreeSize = pow2roundup( NUM_PHOTONS + 1 ) - 1;
    m_photonKdTree = m_context->createBuffer( RT_BUFFER_INPUT );
    m_photonKdTree->setFormat( RT_FORMAT_USER );
    m_photonKdTree->setElementSize( sizeof( Photon ) );
    m_photonKdTree->setSize( m_photonKdTreeSize );
    m_context["photonKdTree"]->set( m_photonKdTree );

#elif ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_UNIFORM_GRID

    m_context["photonsGridCellSize"]->setFloat(0.0f);
    m_context["photonsGridSize"]->setUint(0,0,0);
    m_context["photonsWorldOrigo"]->setFloat(make_float3(0));
    m_photonsHashCells = m_context->createBuffer(RT_BUFFER_OUTPUT);
    m_photonsHashCells->setFormat( RT_FORMAT_UNSIGNED_INT );
    m_photonsHashCells->setSize( NUM_PHOTONS );
    m_hashmapOffsetTable = m_context->createBuffer(RT_BUFFER_OUTPUT);
    m_hashmapOffsetTable->setFormat( RT_FORMAT_UNSIGNED_INT );
    m_hashmapOffsetTable->setSize( PHOTON_GRID_MAX_SIZE+1 );
    m_context["hashmapOffsetTable"]->set( m_hashmapOffsetTable );

#endif
#pragma endregion

    // Volumetric Photon Spheres buffer
#if ENABLE_PARTICIPATING_MEDIA
    {
        m_volumetricPhotonsBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
        m_volumetricPhotonsBuffer->setFormat( RT_FORMAT_USER );
        m_volumetricPhotonsBuffer->setElementSize( sizeof( Photon ) );
        m_volumetricPhotonsBuffer->setSize(NUM_VOLUMETRIC_PHOTONS);
        m_context["volumetricPhotons"]->setBuffer(m_volumetricPhotonsBuffer);

        optix::Geometry photonSpheres = m_context->createGeometry();
        photonSpheres->setPrimitiveCount(NUM_VOLUMETRIC_PHOTONS);
        photonSpheres->setIntersectionProgram(m_context->createProgramFromPTXFile("VolumetricPhotonSphere.cu.ptx", "intersect"));
        photonSpheres->setBoundingBoxProgram(m_context->createProgramFromPTXFile("VolumetricPhotonSphere.cu.ptx", "boundingBox"));

        optix::Material material = m_context->createMaterial();
        material->setAnyHitProgram(RayType::VOLUMETRIC_RADIANCE, m_context->createProgramFromPTXFile("VolumetricPhotonSphereRadiance.cu.ptx", "anyHitRadiance"));
        optix::GeometryInstance volumetricPhotonSpheres = m_context->createGeometryInstance( photonSpheres, &material, &material+1 );
        volumetricPhotonSpheres["photonsBuffer"]->setBuffer(m_volumetricPhotonsBuffer);

        m_volumetricPhotonsRoot = m_context->createGeometryGroup();
        m_volumetricPhotonsRoot->setChildCount(1);
        optix::Acceleration m_volumetricPhotonSpheresAcceleration = m_context->createAcceleration("MedianBvh", "Bvh");
        m_volumetricPhotonsRoot->setAcceleration(m_volumetricPhotonSpheresAcceleration);
        m_volumetricPhotonsRoot->setChild(0, volumetricPhotonSpheres);
        m_context["volumetricPhotonsRoot"]->set(m_volumetricPhotonsRoot);
    }

    // Clear Volumetric Photons Program
    {
        Program program = m_context->createProgramFromPTXFile( "VolumetricPhotonInitialize.cu.ptx", "kernel" );
        m_context->setRayGenerationProgram(OptixEntryPoint::PPM_CLEAR_VOLUMETRIC_PHOTONS_PASS, program );
    }
#endif

    // Indirect Radiance Estimation Buffer
    m_indirectRadianceBuffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, m_width, m_height );
    m_context["indirectRadianceBuffer"]->set( m_indirectRadianceBuffer );
    
    // Indirect Radiance Estimation Program
    {
        Program program = m_context->createProgramFromPTXFile( "IndirectRadianceEstimation.cu.ptx", "kernel" );
        m_context->setRayGenerationProgram(OptixEntryPoint::PPM_INDIRECT_RADIANCE_ESTIMATION_PASS, program );
    }

    // Direct Radiance Estimation Buffer
    m_directRadianceBuffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, m_width, m_height );
    m_context["directRadianceBuffer"]->set( m_directRadianceBuffer );

    // Direct Radiance Estimation Program
    {
        Program program = m_context->createProgramFromPTXFile( "DirectRadianceEstimation.cu.ptx", "kernel" );
        m_context->setRayGenerationProgram(OptixEntryPoint::PPM_DIRECT_RADIANCE_ESTIMATION_PASS, program );
    }

    // Output Buffer
    {
        // vmarz TODO use RT_FORMAT_FLOAT4 for optimal performance (see performance notes in programming guide)
        m_outputBuffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, m_width, m_height );
        m_context["outputBuffer"]->set(m_outputBuffer);
        m_context["outputBufferId"]->setInt(m_outputBuffer->getId());
    }

    // Output Program
    {
        Program program = m_context->createProgramFromPTXFile( "Output.cu.ptx", "kernel" );
        m_context->setRayGenerationProgram(OptixEntryPoint::PPM_OUTPUT_PASS, program );
    }
    

    // VCM initialization
    m_vcmUseVC = true;
    m_vcmUseVM = false;
    m_context["vcmUseVC"]->setInt(m_vcmUseVC);
    m_context["vcmUseVM"]->setInt(m_vcmUseVM);
    
    // used to scale camera_u and camera_v
    m_context["pixelSizeFactor"]->setFloat(1.0f, 1.0f);

#if VCM_UNIFORM_VERTEX_SAMPLING
    m_lightPassLaunchWidth = VCM_LIGHT_PASS_LAUNCH_WIDTH;
    m_lightPassLaunchHeight = VCM_LIGHT_PASS_LAUNCH_HEIGHT;
#else
    m_lightPassLaunchWidth = m_width;
    m_lightPassLaunchHeight = m_height;
#endif

    // light vertex buffer
    m_lightVertexBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
    m_lightVertexBuffer->setFormat( RT_FORMAT_USER );
    m_lightVertexBuffer->setElementSize( sizeof( LightVertex ) );
    m_lightVertexBuffer->setSize( 1u );
    m_context["lightVertexBuffer"]->set(m_lightVertexBuffer);
    m_context["lightVertexBufferId"]->setInt(m_lightVertexBuffer->getId());
    m_context["lightSubpathCount"]->setUint(m_lightPassLaunchWidth * m_lightPassLaunchHeight);

    // VCM light vertex buffer index counter buffer
    m_lightVertexBufferIndexBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, 1u);
    m_context["lightVertexBufferIndexBuffer"]->set(m_lightVertexBufferIndexBuffer);
    m_context["lightVertexBufferIndexBufferId"]->setInt(m_lightVertexBufferIndexBuffer->getId());

    optix::uint* bufferHost = static_cast<optix::uint*>(m_lightVertexBufferIndexBuffer->map());
    memset(bufferHost, 0, sizeof(optix::uint));
    m_lightVertexBufferIndexBuffer->unmap();

    // Size is set in light pass length estimate pass
    m_lightSubpathVertexCountBuffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, 0u );
    m_context["lightSubpathVertexCountBuffer"]->set(m_lightSubpathVertexCountBuffer);
    m_context["lightSubpathVertexCountBufferId"]->setInt(m_lightSubpathVertexCountBuffer->getId());

    m_context["lightVertexCountEstimatePass"]->setUint(1u);
    m_context["vcmNumlightVertexConnections"]->setUint(VCM_NUM_LIGHT_PATH_CONNECTIONS);

#if !VCM_UNIFORM_VERTEX_SAMPLING
    m_lightSubpathVertexIndexBuffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT,
        m_lightPassLaunchWidth * m_lightPassLaunchHeight, VCM_MAX_PATH_LENGTH-1 );
    m_context["lightSubpathVertexIndexBuffer"]->set(m_lightSubpathVertexIndexBuffer);
    m_context["lightSubpathVertexIndexBufferId"]->setInt(m_lightSubpathVertexIndexBuffer->getId());
#endif

    // VCM programs
    {
        Program generatorProgram = m_context->createProgramFromPTXFile( "VCMLightPass.cu.ptx", "lightPass" );
        Program exceptionProgram = m_context->createProgramFromPTXFile( "VCMLightPass.cu.ptx", "exception" );
        Program missProgram = m_context->createProgramFromPTXFile( "VCMLightPass.cu.ptx", "miss");
        m_context->setRayGenerationProgram(OptixEntryPoint::VCM_LIGHT_PASS, generatorProgram);
        m_context->setExceptionProgram(OptixEntryPoint::VCM_LIGHT_PASS, exceptionProgram);
        m_context->setMissProgram(RayType::LIGHT_VCM, missProgram);
    }

    {
        Program generatorProgram = m_context->createProgramFromPTXFile( "VCMCameraPass.cu.ptx", "cameraPass" );
        Program exceptionProgram = m_context->createProgramFromPTXFile( "VCMCameraPass.cu.ptx", "exception" );
        Program missProgram = m_context->createProgramFromPTXFile( "VCMCameraPass.cu.ptx", "miss");
        m_context->setRayGenerationProgram(OptixEntryPoint::VCM_CAMERA_PASS, generatorProgram);
        m_context->setExceptionProgram(OptixEntryPoint::VCM_CAMERA_PASS, exceptionProgram);
        m_context->setMissProgram(RayType::CAMERA_VCM, missProgram);
    }

    // Random state buffer (must be large enough to give states to both photons and image pixels)
    m_randomStatesBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT|RT_BUFFER_GPU_LOCAL);
    m_randomStatesBuffer->setFormat( RT_FORMAT_USER );
    m_randomStatesBuffer->setElementSize( sizeof( RandomState ) );
    m_randomStatesBuffer->setSize( std::max(PHOTON_LAUNCH_WIDTH, m_width), std::max(PHOTON_LAUNCH_HEIGHT, m_height) );
    m_context["randomStates"]->set(m_randomStatesBuffer);
    
    // Light sources buffer
    m_lightBuffer = m_context->createBuffer(RT_BUFFER_INPUT);
    m_lightBuffer->setFormat(RT_FORMAT_USER);
    m_lightBuffer->setElementSize(sizeof(Light));
    m_lightBuffer->setSize(1);
    m_context["lights"]->set( m_lightBuffer );
    m_context["lightsBufferId"]->setInt(m_lightBuffer->getId());

    // Debug buffers
    createGpuDebugBuffers();

#if ENABLE_RENDER_DEBUG_EXCEPTIONS
    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(10000000u); 
    m_context->setExceptionEnabled(RTexception::RT_EXCEPTION_ALL , true); // vmarz: only stack overflow exception enabled by default
#endif
    
    m_initialized = true;

    //printf("Num CPU threads: %d\n", m_context->getCPUNumThreads());
    //printf("GPU paging active: %d\n", m_context->getGPUPagingActive());
    //printf("Enabled devices count: %d\n", m_context->getEnabledDeviceCount());
    //printf("Get devices count: %d\n", m_context->getDeviceCount());
    //printf("Used host memory: %d\n", m_context->getUsedHostMemory());
    //printf("Sizeof Photon %d\n", sizeof(Photon));
}



void OptixRenderer::initDevice(const ComputeDevice & device)
{
    // Set OptiX device as given by ComputeDevice::getDeviceId (Cuda ordinal)

    unsigned int deviceCount = m_context->getDeviceCount();
    int deviceOptixOrdinal = -1;
    for(unsigned int index = 0; index < deviceCount; ++index)
    {
        int cudaDeviceOrdinal;
        if(RTresult code = rtDeviceGetAttribute(index, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(int), &cudaDeviceOrdinal))
            throw Exception::makeException(code, 0);

        if(cudaDeviceOrdinal == device.getDeviceId())
        {
            deviceOptixOrdinal = index;
        }
    }

    m_optixDeviceOrdinal = deviceOptixOrdinal;

    if(deviceOptixOrdinal >= 0)
    {
        m_context->setDevices(&deviceOptixOrdinal, &deviceOptixOrdinal+1);
    }
    else
    {
        throw std::exception("Did not find OptiX device Number for given device. OptiX may not support this device.");
    }
}



void OptixRenderer::initScene( IScene & scene )
{
    if(!m_initialized)
    {
        throw std::exception("Cannot initialize scene before OptixRenderer.");
    }

    const QVector<Light> & lights = scene.getSceneLights();
    if(lights.size() == 0)
    {
        throw std::exception("No lights exists in this scene.");
    }

#if ENABLE_MESH_HITS_COUNTING
    int sceneNMeshes = scene.getNumMeshes();
    m_context["sceneNMeshes"]->setInt(sceneNMeshes);
    optix::Buffer hitsPerMeshBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, sceneNMeshes);
    unsigned int* bufferHost = (unsigned int*)hitsPerMeshBuffer->map();
    memset(bufferHost, 0, sizeof(unsigned int) * sceneNMeshes);
    hitsPerMeshBuffer->unmap();
    m_context["hitsPerMeshBuffer"]->setBuffer(hitsPerMeshBuffer);
#endif

    try
    {
        m_lightVertexCountEstimated = false;

        m_sceneRootGroup = scene.getSceneRootGroup(m_context);
        m_context["sceneRootObject"]->set(m_sceneRootGroup);
        m_sceneAABB = scene.getSceneAABB();
        Sphere sceneBoundingSphere = m_sceneAABB.getBoundingSphere();
        m_context["sceneBoundingSphere"]->setUserData(sizeof(Sphere), &sceneBoundingSphere);

        // Add the lights from the scene to the light buffer

        m_lightBuffer->setSize(lights.size());
        Light* lights_host = (Light*)m_lightBuffer->map();
        memcpy(lights_host, scene.getSceneLights().constData(), sizeof(Light)*lights.size());
        m_lightBuffer->unmap();

        compile();

    }
    catch(const optix::Exception & e)
    {
        m_initialized = false;
        QString error = QString("An OptiX error occurred when initializing scene: %1").arg(e.getErrorString().c_str());
        throw std::exception(error.toLatin1().constData());
    }
}



void OptixRenderer::compile()
{
    try
    {
        printf("Context validation... ");
        m_context->validate();
        printf("Done\nContext compilation... ");
        m_context->compile();
        printf("Done\n");
    }
    catch(const Exception& e)
    {
        throw e;
    }
}



void OptixRenderer::renderNextIteration(unsigned long long iterationNumber, unsigned long long localIterationNumber,
                                        float PPMRadius, bool createOutput, const RenderServerRenderRequestDetails & details)
{
#if ENABLE_RENDER_DEBUG_OUTPUT
    printf("----------------------- %d Local: %d\n", iterationNumber, localIterationNumber);
#endif
    if(!m_initialized)
    {
        throw std::exception("Traced before OptixRenderer was initialized.");
    }

    char buffer[40];
    sprintf(buffer, "OptixRenderer::Trace Iteration %d", iterationNumber);
    nvtx::ScopedRange r(buffer);

#if ENABLE_MESH_HITS_COUNTING
    // print scene meshes count
    int sceneNMeshes = m_context["sceneNMeshes"]->getInt();
    optix::Buffer hitsPerMeshBuffer = m_context["hitsPerMeshBuffer"]->getBuffer();
    unsigned int* bufferHost = (unsigned int*)hitsPerMeshBuffer->map();
    memset(bufferHost, 0, sizeof(unsigned int) * sceneNMeshes);
    hitsPerMeshBuffer->unmap();
#endif

    try
    {
        // If the width and height of the current render request has changed, we must resize buffers
        if(details.getWidth() != m_width || details.getHeight() != m_height)
        {
            this->resizeBuffers(details.getWidth(), details.getHeight());
        }

        // zero out output buf before first iteration
        if (localIterationNumber == 0)
        {
            float3* buffer = reinterpret_cast<float3*>( m_outputBuffer->map() );
            memset(buffer, 0, sizeof(optix::float3) * m_width * m_height);
            m_outputBuffer->unmap();
        }

        const Camera & camera = details.getCamera();
        const RenderMethod::E renderMethod = details.getRenderMethod();

        double traceStartTime;
        sutilCurrentTime(&traceStartTime);
    
        double t0, t1;

        m_context["camera"]->setUserData( sizeof(Camera), &camera );
        m_context["iterationNumber"]->setFloat( static_cast<float>(iterationNumber));
        m_context["localIterationNumber"]->setUint((unsigned int)localIterationNumber);

        if (renderMethod == RenderMethod::PATH_TRACING)
        {
            m_context["ptDirectLightSampling"]->setInt(1);
            nvtx::ScopedRange r("OptixEntryPoint::PT_RAYTRACE_PASS");
            sutilCurrentTime( &t0 );
            m_context->launch( OptixEntryPoint::PT_RAYTRACE_PASS,
                static_cast<unsigned int>(m_width),
                static_cast<unsigned int>(m_height) );
            sutilCurrentTime( &t1 );
        }
        else if (renderMethod == RenderMethod::PROGRESSIVE_PHOTON_MAPPING)
        {          
#pragma region PROGRESSIVE PHOTON MAPPING
            // Trace viewing rays
            {
                nvtx::ScopedRange r("OptixEntryPoint::RAYTRACE_PASS");
                sutilCurrentTime( &t0 );
                m_context->launch( OptixEntryPoint::PPM_RAYTRACE_PASS,
                    static_cast<unsigned int>(m_width),
                    static_cast<unsigned int>(m_height) );
                sutilCurrentTime( &t1 );
            }

            // Update PPM Radius for next photon tracing pass
            const float ppmAlpha = details.getPPMAlpha();
            m_context["ppmAlpha"]->setFloat(ppmAlpha);
            const float ppmRadiusSquared = PPMRadius*PPMRadius;
            m_context["ppmRadius"]->setFloat(PPMRadius);
            m_context["ppmRadiusSquared"]->setFloat(ppmRadiusSquared);
            const float ppmRadiusSquaredNew = ppmRadiusSquared*(iterationNumber+ppmAlpha)/float(iterationNumber+1);
            m_context["ppmRadiusSquaredNew"]->setFloat(ppmRadiusSquaredNew);


#if ENABLE_PARTICIPATING_MEDIA
            m_context["volumetricRadius"]->setFloat(0.033/0.033*PPMRadius);
            
            // Clear volume photons
            {
                nvtx::ScopedRange r( "OptixEntryPoint::PPM_CLEAR_VOLUMETRIC_PHOTONS_PASS" );
                m_context->launch( OptixEntryPoint::PPM_CLEAR_VOLUMETRIC_PHOTONS_PASS, NUM_VOLUMETRIC_PHOTONS);
            }
#endif
            // Set up the uniform grid bounds
            #if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_STOCHASTIC_HASH
            {
                nvtx::ScopedRange r("initializeStochasticHashPhotonMap()");
                initializeStochasticHashPhotonMap(PPMRadius);
            }
            #endif

            // Photon Tracing
            {
                nvtx::ScopedRange r( "OptixEntryPoint::PHOTON_PASS" );
                m_context->launch( OptixEntryPoint::PPM_PHOTON_PASS,
                    static_cast<unsigned int>(PHOTON_LAUNCH_WIDTH),
                    static_cast<unsigned int>(PHOTON_LAUNCH_HEIGHT) );

                float totalEmitted = (iterationNumber+1)*EMITTED_PHOTONS_PER_ITERATION;
                m_context["totalEmitted"]->setFloat( static_cast<float>(totalEmitted));
            }

            debugOutputPhotonTracing();

            // Create Photon Map
            {
                nvtx::ScopedRange r( "Creating photon map" );
#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_KD_TREE_CPU
                createPhotonKdTreeOnCPU();
#elif ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_UNIFORM_GRID
                createUniformGridPhotonMap(PPMRadius);
#endif
            }

#if ENABLE_PARTICIPATING_MEDIA
            // Rebuild the volumetric photons BVH
            {
                double t0, t1;
                sutilCurrentTime( &t0 );
                m_volumetricPhotonsRoot->getAcceleration()->markDirty();
                m_context->launch(OptixEntryPoint::PPM_RAYTRACE_PASS, 0, 0);
                sutilCurrentTime( &t1);
                if(iterationNumber % 20 == 0 && iterationNumber < 100)
                {
                    printf("Rebuilt volumetric photons (%d photons) in %.4f.\n", NUM_VOLUMETRIC_PHOTONS, t1-t0);
                }
            }
#endif

            // Transfer any data from the photon acceleration structure build to the GPU (trigger an empty launch)
            {
                nvtx::ScopedRange r("Transfer photon map to GPU");
                m_context->launch(OptixEntryPoint::PPM_INDIRECT_RADIANCE_ESTIMATION_PASS,
                    0, 0);
            }
    
            // PPM Indirect Estimation (using the photon map)
            {
                nvtx::ScopedRange r("OptixEntryPoint::INDIRECT_RADIANCE_ESTIMATION");
                m_context->launch(OptixEntryPoint::PPM_INDIRECT_RADIANCE_ESTIMATION_PASS,
                    m_width, m_height);
            }

            // Direct Radiance Estimation
            {
                nvtx::ScopedRange r("OptixEntryPoint::PPM_DIRECT_RADIANCE_ESTIMATION_PASS");
                m_context->launch(OptixEntryPoint::PPM_DIRECT_RADIANCE_ESTIMATION_PASS,
                    m_width, m_height);
            }

            // Combine indirect and direct buffers in the output buffer
            nvtx::ScopedRange r("OptixEntryPoint::PPM_OUTPUT_PASS");
            m_context->launch(OptixEntryPoint::PPM_OUTPUT_PASS,
                m_width, m_height);
#pragma endregion PROGRESSIVE PHOTON MAPPING

        }
        else if (renderMethod == RenderMethod::VCM_BIDIRECTIONAL_PATH_TRACING)
        {
            const unsigned int cameraSubPathCount = m_width * m_height;
            const unsigned int lightSubPathCount = m_lightPassLaunchWidth * m_lightPassLaunchHeight;
            const unsigned int lightSubpathsConnected = VCM_UNIFORM_VERTEX_SAMPLING ? lightSubPathCount : 1.f;  // nVC
            const unsigned int lightSubpathsMerged = lightSubPathCount;                                         // nVM
            const float ppmRadiusSquared = PPMRadius*PPMRadius; // vmarz TODO change radius reduction scheme
                
            // 1/(PI*r*r) from VM path pdf [tech. rep. (10)], 1/lightSubPathCount from MIS estimator (not weight) [tech. rep. (11)]
            // 1 / (radius^2 * PI) comes density estimation kernel Kr [VCM paper (18)], in this case simply scales by area
            // used to normalize VM estimator
            const float vmNormalizationFactor = 1.f / (ppmRadiusSquared * M_PIf * lightSubPathCount);
                
            // MIS weight constant [tech. rep. (20)]
            // etaVCM = (nVM / nVC) * PI * r2
            const float etaVCM = (float(lightSubpathsMerged) / lightSubpathsConnected) * M_PIf * ppmRadiusSquared;
            const float misVmWeightFactor = m_vcmUseVM ? vcmMis(etaVCM)       : 0.f;
            const float misVcWeightFactor = m_vcmUseVC ? vcmMis(1.f / etaVCM) : 0.f;

            m_context["vmNormalizationFactor"]->setFloat(vmNormalizationFactor);
            m_context["misVmWeightFactor"]->setFloat(misVmWeightFactor);
            m_context["misVcWeightFactor"]->setFloat(misVcWeightFactor);

            // Estimate average light subpath length and initialize vertex buffer with appropriate size
            if (!m_lightVertexCountEstimated)
            {
                unsigned int estimateWidth = m_lightPassLaunchWidth;
                unsigned int estimateHeight = m_lightPassLaunchHeight;

                m_lightSubpathVertexCountBuffer->setSize(estimateWidth * estimateHeight);

                // Transfer any data to the GPU (trigger an empty launch)
                dbgPrintf("VCM: init empty launch\n");
                m_context->launch( OptixEntryPoint::VCM_LIGHT_PASS, 0u, 0u);
                dbgPrintf("Available device memory MB %f\n", m_context->getAvailableDeviceMemory(0u) / 1000000.f);

                // Estimate light subpath length to initialize light vertex cache (LVC)
                {
                    dbgPrintf("OptixEntryPoint::VCM_LIGHT_PASS subpath length estimate launch dim %u x %u\n", estimateWidth, estimateHeight);
                    m_context["maxPathLen"]->setUint(VCM_MAX_PATH_LENGTH);
                    m_context["lightVertexCountEstimatePass"]->setUint(1u);
                    nvtx::ScopedRange r("OptixEntryPoint::VCM_LIGHT_PASS");
                    sutilCurrentTime( &t0 );
                    m_context->launch( OptixEntryPoint::VCM_LIGHT_PASS, estimateWidth, estimateHeight );
                    sutilCurrentTime( &t1 );
                    dbgPrintf("LVC size estimate launch time: %.4f.\n", t1-t0);
                }

                // get average stored vertex count
                optix::uint* buffer_Host = static_cast<optix::uint*>(m_lightSubpathVertexCountBuffer->map());
                const unsigned int subpathEstimateCount = estimateWidth * estimateHeight;
                unsigned long long sumStoredVerts = 0;
                unsigned int maxPathVerts = 0;

                for(int i = 0; i < subpathEstimateCount ; i++)
                {
                    unsigned int count = buffer_Host[i];
                    if (maxPathVerts < count) maxPathVerts = count;
                    if (0 < count) sumStoredVerts += count;
                }
                m_lightSubpathVertexCountBuffer->unmap();

                float avgSubpathVerts = float(sumStoredVerts) / subpathEstimateCount;
                dbgPrintf("LVC estimate. paths: %u  vertices: %llu  avgStoredVertices: %f  maxPathVertices: %u\n",
                    subpathEstimateCount, sumStoredVerts, avgSubpathVerts, maxPathVerts);

                // Init light vertex buffer based on average length.
                // Estimate currently is inaccurate in case when with distance light source so add big safety margin,
                // if it happens to be bigger than allowed length then just use that one instead
                avgSubpathVerts = std::min(avgSubpathVerts / 0.5f, float(VCM_MAX_PATH_LENGTH-1));
                const unsigned int vertBufSize = (lightSubPathCount * avgSubpathVerts);
                m_lightVertexBuffer->setSize(vertBufSize);
                dbgPrintf("Vertex buffer size set to: %u \n", vertBufSize);
                
                // Resize vertex count buffer for regular light pass
                m_lightSubpathVertexCountBuffer->setSize(m_lightPassLaunchWidth * m_lightPassLaunchHeight);

#if !VCM_UNIFORM_VERTEX_SAMPLING
                m_lightSubpathVertexIndexBuffer->setSize(m_lightPassLaunchWidth * m_lightPassLaunchHeight, VCM_MAX_PATH_LENGTH-1);
                //m_context["maxPathLen"]->setUint(VCM_MAX_PATH_LENGTH); increase?
#else
                // vertex pick pdf
                const float vertexPickPdf = avgSubpathLength / sumPathLengths;
                dbgPrintf("VCM: Vertex pick pdf        %f \n", vertexPickPdf);
                m_context["vertexPickPdf"]->setFloat(vertexPickPdf);
                m_context["averageLightSubpathLength"]->setFloat(avgSubpathLength);
#endif                
                m_lightVertexCountEstimated = true;
                m_context["lightVertexCountEstimatePass"]->setUint(0u);

                dbgPrintf("VCM: cameraSubPathCount     %u \n", cameraSubPathCount);
                dbgPrintf("VCM: lightSubpathsMerged    %u \n", lightSubpathsMerged);
                dbgPrintf("VCM: lightSubpathsConnected %u \n", lightSubpathsConnected);
                dbgPrintf("VCM: ppmRadius              %.10f \n", PPMRadius);
                dbgPrintf("VCM: etaVCM                 %.10f \n", etaVCM);
                dbgPrintf("VCM: misVmWeightFactor      %.10f \n", misVmWeightFactor);
                dbgPrintf("VCM: misVcWeightFactor      %.10f \n", misVcWeightFactor);
                dbgPrintf("VCM: vmNormalizationFactor  %.10f \n", vmNormalizationFactor);
            }

            // Reset buffer index counter (not used it length estimate pass)
            optix::uint* bufferHost = static_cast<optix::uint*>(m_lightVertexBufferIndexBuffer->map());
            memset(bufferHost, 0, sizeof(optix::uint));
            m_lightVertexBufferIndexBuffer->unmap();

            // Light pass
            { 
                nvtx::ScopedRange r("OptixEntryPoint::VCM_LIGHT_PASS");
                sutilCurrentTime( &t0 );
                m_context->launch( OptixEntryPoint::VCM_LIGHT_PASS, m_lightPassLaunchWidth, m_lightPassLaunchHeight );
                sutilCurrentTime( &t1 );
                dbgPrintf("Light pass launch time: %.4f.\n", t1-t0);
            }

            // Camera pass
            { 
                nvtx::ScopedRange r("OptixEntryPoint::VCM_CAMERA_PASS");
                sutilCurrentTime( &t0 );
                m_context->launch( OptixEntryPoint::VCM_CAMERA_PASS, m_width, m_height );
                sutilCurrentTime( &t1 );
                dbgPrintf("Camera pass launch time: %.4f.\n", t1-t0);
            }
        }

        double end;
        sutilCurrentTime( &end );
        double traceTime = end-traceStartTime;

#if ENABLE_MESH_HITS_COUNTING
        // print scene meshes count
        int sceneNMeshes = m_context["sceneNMeshes"]->getInt();
        optix::Buffer hitsPerMeshBuffer = m_context["hitsPerMeshBuffer"]->getBuffer();
        unsigned int* bufferHost = (unsigned int*)hitsPerMeshBuffer->map();
        for(int i = 0; i < sceneNMeshes; i++)
        {
            if(bufferHost[i] > 0)
                printf("hitsPerMesh [%i] = %u\n", i, bufferHost[i]);
        }
        hitsPerMeshBuffer->unmap();
#endif
    }
    catch(const optix::Exception & e)
    {
        QString error = QString("An OptiX error occurred: %1").arg(e.getErrorString().c_str());
        throw std::exception(error.toLatin1().constData());
    }
}

static inline unsigned int max(unsigned int a, unsigned int b)
{
    return a > b ? a : b;
}

void OptixRenderer::resizeBuffers(unsigned int width, unsigned int height)
{
    m_outputBuffer->setSize( width, height );
    m_raytracePassOutputBuffer->setSize( width, height );
    m_directRadianceBuffer->setSize( width, height );
    m_indirectRadianceBuffer->setSize( width, height );
    m_randomStatesBuffer->setSize(max(PHOTON_LAUNCH_WIDTH, (unsigned int)width), max(PHOTON_LAUNCH_HEIGHT,  (unsigned int)height));
    initializeRandomStates();
    m_width = width;
    m_height = height;

#if !VCM_UNIFORM_VERTEX_SAMPLING // when using uniform sampling then there is no need to align with output buffer size
    m_lightPassLaunchWidth = m_width;
    m_lightPassLaunchHeight = m_height;
    m_lightSubpathVertexIndexBuffer->setSize(m_lightPassLaunchWidth * m_lightPassLaunchHeight, VCM_MAX_PATH_LENGTH-1);
    m_context["lightSubpathCount"]->setUint(m_lightPassLaunchWidth * m_lightPassLaunchHeight);
#endif
    // used to scale camera_u and camera_v for VCM
    m_context["pixelSizeFactor"]->setFloat(1.0f / width, 1.0f / height);
    m_lightVertexCountEstimated = false;
}

unsigned int OptixRenderer::getWidth() const
{
    return m_width;
}

unsigned int OptixRenderer::getHeight() const
{
    return m_height;
}

void OptixRenderer::getOutputBuffer( void* data )
{
    void* buffer = reinterpret_cast<void*>( m_outputBuffer->map() );
    memcpy(data, buffer, getScreenBufferSizeBytes());
    m_outputBuffer->unmap();
}

unsigned int OptixRenderer::getScreenBufferSizeBytes() const
{
    return m_width*m_height*sizeof(optix::float3);
}

void OptixRenderer::debugOutputPhotonTracing()
{
#if ENABLE_RENDER_DEBUG_OUTPUT
    printf("Grid size: %d %d %d. Cellsize: %.4f\n", m_gridSize.x, m_gridSize.y, m_gridSize.z, m_context["photonsGridCellSize"]->getFloat());
    {
        optix::Buffer buffer = m_context["debugPhotonPathLengthBuffer"]->getBuffer();
        unsigned int* buffer_Host = (unsigned int*)buffer->map();
        unsigned long long sumPaths = 0;
        unsigned int numZero = 0;
        for(int i = 0; i < PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT; i++)
        {
            sumPaths += buffer_Host[i];
            if(buffer_Host[i] == 0)
            {
                numZero++;
            }
        }
        buffer->unmap();
        double averagePathLength = double(sumPaths)/(PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT);
        double percentageZero = 100*double(numZero)/(PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT);
        printf("  Average photonprd path length: %.4f (Paths with 0: %.4f%%)\n", averagePathLength, percentageZero);
    }

    {
        optix::Buffer buffer = m_context["debugIndirectRadianceCellsVisisted"]->getBuffer();
        unsigned int* buffer_Host = (unsigned int*)buffer->map();
        unsigned long long sumVisited = 0;
        unsigned int numNotZero = 0;
        for(int i = 0; i < m_width*m_height; i++)
        {
            if(buffer_Host[i] > 0)
            {
                sumVisited += buffer_Host[i];
                numNotZero++;
            }
        }
        buffer->unmap();
        double visitedAvg = 0 < numNotZero ? double(sumVisited)/(numNotZero) : 0;
        printf("  Average cells visited during indirect estimation  (per pixel): %.4f\n", visitedAvg);
    }

    {
        optix::Buffer buffer = m_context["debugIndirectRadiancePhotonsVisisted"]->getBuffer();
        unsigned int* buffer_Host = (unsigned int*)buffer->map();
        unsigned long long sumVisited = 0;
        unsigned int numNotZero = 0;
        for(int i = 0; i < m_width*m_height; i++)
        {
            if(buffer_Host[i] > 0)
            {
                sumVisited += buffer_Host[i];
                numNotZero++;
            }
        }
        buffer->unmap();
        double visitedAvg = 0 < numNotZero ? double(sumVisited)/(numNotZero) : 0;
        printf("  Average photons visited during indirect estimation (per pixel): %.4f\n", visitedAvg);
    }

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_STOCHASTIC_HASH
    {
        const unsigned int hashTableSize = NUM_PHOTONS;
        optix::Buffer buffer = m_context["photonsHashTableCount"]->getBuffer();
        unsigned int* buffer_Host = (unsigned int*)buffer->map();
        unsigned int numFilled = 0;
        float sumCollisions = 0;
        for(int i = 0; i < hashTableSize; i++)
        {
            if(buffer_Host[i] > 0)
            {
                numFilled++;
                sumCollisions += buffer_Host[i];
            }
        }
        buffer->unmap();
        double fillRate = 100*double(numFilled)/hashTableSize;
        double averageCollisions = double(sumCollisions)/numFilled;
        printf("  Table size %d Filled: %d fill%%: %.4f\n  Uniform grid collisions (in filled cells): %.4f\n", hashTableSize, numFilled, fillRate, averageCollisions);
    }
#endif
#endif
}

void OptixRenderer::createGpuDebugBuffers()
{
#if ENABLE_RENDER_DEBUG_OUTPUT
    optix::Buffer debugPhotonPathLengthBuffer = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, PHOTON_LAUNCH_WIDTH, PHOTON_LAUNCH_HEIGHT);
    m_context["debugPhotonPathLengthBuffer"]->setBuffer(debugPhotonPathLengthBuffer);
    optix::Buffer debugIndirectRadianceCellsVisisted = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, 2000, 2000);
    m_context["debugIndirectRadianceCellsVisisted"]->setBuffer(debugIndirectRadianceCellsVisisted);
    optix::Buffer debugIndirectRadiancePhotonsVisisted = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, 2000, 2000);
    m_context["debugIndirectRadiancePhotonsVisisted"]->setBuffer(debugIndirectRadiancePhotonsVisisted);
#endif
}
