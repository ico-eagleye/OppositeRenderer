/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Stian Pedersen
 *                Valdis Vilcans
*/

#pragma once

#define PPM_X         ( 1 << 0 )
#define PPM_Y         ( 1 << 1 )
#define PPM_Z         ( 1 << 2 )
#define PPM_LEAF      ( 1 << 3 )
#define PPM_NULL      ( 1 << 4 )

#define ACCELERATION_STRUCTURE_UNIFORM_GRID 0
#define ACCELERATION_STRUCTURE_KD_TREE_CPU 1
#define ACCELERATION_STRUCTURE_STOCHASTIC_HASH 2
#define ACCELERATION_STRUCTURE (ACCELERATION_STRUCTURE_UNIFORM_GRID)

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_STOCHASTIC_HASH
#define MAX_PHOTONS_DEPOSITS_PER_EMITTED 1
#else
#define MAX_PHOTONS_DEPOSITS_PER_EMITTED 4
#endif

#define ENABLE_PARTICIPATING_MEDIA 0
#define ENABLE_RENDER_DEBUG_EXCEPTIONS 1
#define ENABLE_RENDER_DEBUG_OUTPUT 1

#define MAX_PHOTON_TRACE_DEPTH (ENABLE_PARTICIPATING_MEDIA?15:7)
#define MAX_RADIANCE_TRACE_DEPTH 9
#define NUM_VOLUMETRIC_PHOTONS 200000
#define PHOTON_TRACING_RR_START_DEPTH 3
#define PATH_TRACING_RR_START_DEPTH 3

#define USE_CHEAP_RANDOM 0

#define RAY_LEN_MIN 0.0001f
#define EPS_COSINE 1e-6f
#define EPS_RAY    1e-3f

#define ENABLE_MESH_HITS_COUNTING 0

#define MAX_OUTPUT_X 2000
#define MAX_OUTPUT_Y 2000

//#define DEBUG_RANDOM_SEED 1645301512
