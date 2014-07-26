/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Stian Pedersen
 *                Valdis Vilcans
*/

#pragma once
#include "math/Vector3.h"
#include "render_engine_export_api.h"

class Light
{
public:
    enum LightType {AREA, POINT, SPOT};

#ifndef __CUDACC__
    RENDER_ENGINE_EXPORT_API Light(){};
    RENDER_ENGINE_EXPORT_API Light(Vector3 power, Vector3 position, Vector3 v1, Vector3 v2);
    RENDER_ENGINE_EXPORT_API Light(Vector3 power, Vector3 position);
    RENDER_ENGINE_EXPORT_API Light(Vector3 power, Vector3 position, Vector3 direction, float angle);

#endif

    optix::float3 power;
    optix::float3 position;
    optix::float3 v1;
    optix::float3 v2;
    bool isDelta;		// Whether the light has delta function (point, directional) or not (area)
    bool isFinite;		// Whether the light has a finite extent (area, point) or not (directional, env. map)

    float inverseArea;
    union
    {
        float area; // area
        float angle; // spot
    };

    union
    {
        optix::float3 Lemit; // area
        optix::float3 intensity; // spot
    };

    union
    {
        optix::float3 normal; // area
        optix::float3 direction; // spot
    };

    LightType lightType;
};