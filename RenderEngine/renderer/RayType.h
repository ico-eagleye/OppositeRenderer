/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Stian Pedersen
 *                Valdis Vilcans
*/

#pragma once
namespace RayType
{
    enum E
    {
        RADIANCE,
        PHOTON,
        CAMERA_VCM,
        LIGHT_VCM,
        RADIANCE_IN_PARTICIPATING_MEDIUM,
        PHOTON_IN_PARTICIPATING_MEDIUM,
        VOLUMETRIC_RADIANCE,
        SHADOW,
        PARTICIPATING_MEDIUM_TRANSMISSION,
        NUM_RAY_TYPES
    };
}
