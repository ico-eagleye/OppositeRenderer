/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

struct SubpathPRD
{
  optix::float3 origin;
  optix::float3 direction;
  optix::float3 throughput;
  optix::uint depth;
  uint seed;
  float dVCM;
  float dVC;
  float dVM;
  optix::uint done;
};
