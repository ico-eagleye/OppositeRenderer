/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Stian Pedersen
 *                Valdis Vilcans
 */

#include "Glass.h"
#include "renderer/RayType.h"

bool Glass::m_optixMaterialIsCreated = false;
optix::Material Glass::m_optixMaterial;

Glass::Glass( float indexOfRefraction, const Vector3 & Kr, const Vector3 & Kt )
{
    this->indexOfRefraction = indexOfRefraction;
    this->Kr = Kr;
    this->Kt = Kt;
}

optix::Material Glass::getOptixMaterial(optix::Context & context)
{
    if(!m_optixMaterialIsCreated)
    {
        m_optixMaterial = context->createMaterial();
        optix::Program radianceClosestProgram = context->createProgramFromPTXFile( "Glass.cu.ptx", "closestHitRadiance");
        optix::Program radianceAnyHitProgram = context->createProgramFromPTXFile( "Glass.cu.ptx", "anyHitRadiance");
        optix::Program photonClosestProgram = context->createProgramFromPTXFile( "Glass.cu.ptx", "closestHitPhoton");
        optix::Program photonAnyHitProgram = context->createProgramFromPTXFile( "Glass.cu.ptx", "anyHitPhoton");

        //vcmClosestHitLight

        m_optixMaterial->setClosestHitProgram(RayType::RADIANCE, radianceClosestProgram);
        //m_optixMaterial->setAnyHitProgram(RayType::RADIANCE, radianceAnyHitProgram );
        m_optixMaterial->setClosestHitProgram(RayType::RADIANCE_IN_PARTICIPATING_MEDIUM, radianceClosestProgram);
        //m_optixMaterial->setAnyHitProgram(RayType::RADIANCE_IN_PARTICIPATING_MEDIUM, radianceAnyHitProgram);
        
        m_optixMaterial->setClosestHitProgram(RayType::PHOTON, photonClosestProgram);
        //m_optixMaterial->setAnyHitProgram(RayType::PHOTON, photonAnyHitProgram);
        m_optixMaterial->setClosestHitProgram(RayType::PHOTON_IN_PARTICIPATING_MEDIUM, photonClosestProgram);
       // m_optixMaterial->setAnyHitProgram(RayType::PHOTON_IN_PARTICIPATING_MEDIUM, photonAnyHitProgram);

        m_optixMaterial->setClosestHitProgram(RayType::LIGHT_VCM, context->createProgramFromPTXFile( "Glass.cu.ptx", "vcmClosestHitLight"));
        m_optixMaterial->setClosestHitProgram(RayType::CAMERA_VCM, context->createProgramFromPTXFile( "Glass.cu.ptx", "vcmClosestHitCamera"));

        this->registerMaterialWithShadowProgram(context, m_optixMaterial);
        m_optixMaterialIsCreated = true;
    }
    
    return m_optixMaterial;
}

/*
// Register any material-dependent values to be available in the optix program.
*/
void Glass::registerGeometryInstanceValues(optix::GeometryInstance & instance )
{
    instance["indexOfRefraction"]->setFloat(this->indexOfRefraction);
    //instance["Kd"]->setFloat( 0, 0 , 0 );
    instance["Kr"]->setFloat(this->Kr);
    instance["Kt"]->setFloat(this->Kt);
}
