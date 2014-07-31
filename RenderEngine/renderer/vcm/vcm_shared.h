/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Valdis Vilcans
*/

#pragma once
#include <renderer/device_common.h>

// Applies MIS power
static __host__ RT_FUNCTION float vcmMis(const float & aPdf)
{
    // balance heuristic for now
    return aPdf;
}