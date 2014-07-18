/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "Glossy.h"
#include "renderer/RayType.h"

bool Glossy::m_optixMaterialIsCreated = false;
optix::Material Glossy::m_optixMaterial;

Glossy::Glossy(const Vector3 & Kd, const Vector3 & Ks, const float exponent)
{
    this->m_Kd = Kd;
    this->m_Ks = Ks;
    this->m_exponent = exponent;
}

optix::Material Glossy::getOptixMaterial(optix::Context & context)
{
    if (m_optixMaterialIsCreated)
        return m_optixMaterial;

    m_optixMaterial = context->createMaterial();
    optix::Program radianceProgram = context->createProgramFromPTXFile( "Glossy.cu.ptx", "closestHitRadiance");
    optix::Program photonProgram = context->createProgramFromPTXFile( "Glossy.cu.ptx", "closestHitPhoton");
    m_optixMaterial->setClosestHitProgram(RayType::RADIANCE, radianceProgram);
    m_optixMaterial->setClosestHitProgram(RayType::RADIANCE_IN_PARTICIPATING_MEDIUM, radianceProgram);
    m_optixMaterial->setClosestHitProgram(RayType::PHOTON, photonProgram);
    m_optixMaterial->setClosestHitProgram(RayType::PHOTON_IN_PARTICIPATING_MEDIUM, photonProgram);

    m_optixMaterial->setClosestHitProgram(RayType::LIGHT_VCM, context->createProgramFromPTXFile( "Glossy.cu.ptx", "vcmClosestHitLight"));
    m_optixMaterial->setClosestHitProgram(RayType::CAMERA_VCM, context->createProgramFromPTXFile( "Glossy.cu.ptx", "vcmClosestHitCamera"));
    m_optixMaterial->validate();

    this->registerMaterialWithShadowProgram(context, m_optixMaterial);
    m_optixMaterialIsCreated = true;

    return m_optixMaterial;
}

/*
// Register any material-dependent values to be available in the optix program.
*/
void Glossy::registerGeometryInstanceValues(optix::GeometryInstance & instance )
{
    instance["Kd"]->setFloat(this->m_Kd);
    instance["Ks"]->setFloat(this->m_Ks);
    instance["exponent"]->setFloat(this->m_exponent);
}