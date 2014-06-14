/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "Light.h"
#include <optixu/optixu_math_namespace.h>

Light::Light( Vector3 power, Vector3 position, Vector3 v1, Vector3 v2 )
    : power(power),
    position(position),
    v1(v1),
    v2(v2),
    lightType(LightType::AREA)
{
    optix::float3 crossProduct = optix::cross(v1, v2);
    normal = Vector3(optix::normalize(crossProduct));
    area = length(crossProduct);
    inverseArea = 1.0f/area;
	Lemit = power * inverseArea * M_1_PIf;
	isDelta = false;
	isFinite = true;
}

Light::Light(Vector3 power, Vector3 position)
    : power(power),
    position(position),
    lightType(LightType::POINT)
{
	intensity = power * 0.25f * M_1_PIf;
	isDelta = true;
	isFinite = true;
}

Light::Light( Vector3 power, Vector3 position, Vector3 direction, float angle )
    : power(power), position(position), direction(direction), angle(angle), lightType(LightType::SPOT)
{
    direction = optix::normalize(direction);
	// based on Pharr, Huphreys PBR p.614
	float angleFactor = 1.0f / (1.0f - cosf(angle * 180 * M_1_PIf)); // assume angle in degrees
	intensity = power * 0.25f * M_1_PIf * angleFactor;
	isDelta = true;
	isFinite = true;
}
