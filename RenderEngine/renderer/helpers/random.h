/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once 

#include "config.h"
#include "renderer/RandomState.h"
#include "renderer/device_common.h"
#include "renderer/helpers/helpers.h"
#include <stdint.h>
#include <float.h>

#if USE_CHEAP_RANDOM

/*
The fast/cheap random generation scheme courtesy of 
http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
*/

static uint32_t __host__ __device__ rand_xorshift(uint32_t& state)
{
    state ^= (state << 13);
    state ^= (state >> 17);
    state ^= (state << 5);
    return state;
}

static uint32_t  __host__ __device__ wang_hash(uint32_t seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

static void __host__ __device__ initializeRandomState(RandomState* state, uint32_t seed, uint32_t index)
{
    state[0] = wang_hash(seed+index);
}

// Return a float from [0,1)
static __device__ __inline__ float getRandomUniformFloat( RandomState* state )
{
    float scale = float(0xFFFFFFFF);
    // Clear the last bit to be strictly less than 1
    return float(rand_xorshift(*state) & ~1)/scale;
}

#else

static void __device__ initializeRandomState(RandomState* state, unsigned int seed, unsigned int index)
{
    curand_init(seed+index, 0, 0, state);
}

// Return a float in range [0,1)
static RT_FUNCTION float getRandomUniformFloat( RandomState* state )
{
    // Currand generates values in range (0,1]
    return maxf(curand_uniform(state) - FLT_EPSILON, 0.0f);
}

#endif

// Return a float in range [0,1)
static RT_FUNCTION optix::float2 getRandomUniformFloat2( RandomState* state )
{
    // Currand generates values in range (0,1]
    optix::float2 sample;
    sample.x = getRandomUniformFloat(state);
    sample.y = getRandomUniformFloat(state);
    return sample;
}

// Return a float in range [0,1)
static RT_FUNCTION optix::float3 getRandomUniformFloat3( RandomState* state )
{
    // Currand generates values in range (0,1]
    optix::float3 sample;
    sample.x = getRandomUniformFloat(state);
    sample.y = getRandomUniformFloat(state);
    sample.z = getRandomUniformFloat(state);
    return sample;
}

// <Random number generation used in Optix SDK>
// Generate random unsigned int in [0, 2^24)
static __host__ RT_FUNCTION unsigned int lcg(unsigned int &prev)
{
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

// Generate random float in [0, 1)
static __host__ RT_FUNCTION float rnd(unsigned int &prev)
{
  return ((float) lcg(prev) / (float) 0x01000000);
}

template<unsigned int N>
static __host__ RT_FUNCTION unsigned int tea( unsigned int val0, unsigned int val1 )
{
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for( unsigned int n = 0; n < N; n++ )
    {
        s0 += 0x9e3779b9;
        v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
        v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
    }

    return v0;
}
// </Random number generation used in Optix SDK>