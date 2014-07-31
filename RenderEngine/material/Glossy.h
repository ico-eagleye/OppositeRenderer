/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Valdis Vilcans
*/

#pragma once
#include "Material.h"
#include "math/Vector3.h"

class Glossy : public Material
{
private:
    Vector3 m_Kd;
    Vector3 m_Ks;
    float m_exponent;
    static bool m_optixMaterialIsCreated;
    static optix::Material m_optixMaterial;

public:
    Glossy(const Vector3 & Kd, const Vector3 & Ks, const float exponent);
    virtual optix::Material getOptixMaterial(optix::Context & context);
    virtual void registerGeometryInstanceValues(optix::GeometryInstance & instance);
};