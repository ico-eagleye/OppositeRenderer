/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "config.h"

#if USE_CHEAP_RANDOM
#include <stdint.h>
typedef uint32_t RandomState;
#else
#include <curand_kernel.h>
typedef curandState RandomState;
#endif