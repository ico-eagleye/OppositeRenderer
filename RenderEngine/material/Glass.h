/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#pragma once
#include "Material.h"
#include "math/Vector3.h"

class Glass : public Material
{
private:
    float indexOfRefraction;
    Vector3 Kr;
    Vector3 Kt;
    static bool m_optixMaterialIsCreated;
    static optix::Material m_optixMaterial;
public:
    Glass(float indexOfRefraction, const Vector3 & Kr, const Vector3 & Kt);
    virtual optix::Material getOptixMaterial(optix::Context & context);
    virtual void registerGeometryInstanceValues(optix::GeometryInstance & instance);
};