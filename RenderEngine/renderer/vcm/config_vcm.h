/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

// put in config.h later

#define VCM_UNIFORM_VERTEX_SAMPLING 0

#define DEBUG_SCATTER_MIRROR_REFLECT 0
#define DEBUG_EMIT_DIR_FIXED 0
#define DEBUG_EMIT_DIR optix::normalize(optix::make_float3(0.f, -1.f, 0.f))