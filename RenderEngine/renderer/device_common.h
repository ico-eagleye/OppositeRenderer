/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Valdis Vilcans
*/

#pragma  once

// Optix inlines all functions, Cuda compiles sometimes fails to inline many parameter functions with __inline__ hint
// so this is precaution macro
#define RT_FUNCTION __forceinline__ __device__