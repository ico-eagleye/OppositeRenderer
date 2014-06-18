This project is made reproduce OptiX hangs/crashes discussed in this Optix forum thread
https://devtalk.nvidia.com/default/topic/751906/optix/weird-ray-generation-hang-really-simple-code-/

BUILDING:
1. Define OPTIX_PATH environment variable pointing to OptiX SDK. 
2. Define CUDA_USE_VER environment variable as "5.5" or "6.0". It controls which Cuda build customization .props
and .targets files will be imported into projects.
3. Build ContextTest.sln

Possible to use also OPTIX_USE_VER quickly switch between different OptiX SDKs, but then need to define
their path variables (e.g. OPTIX_PATH_V3_5_1 similarly to Cuda vars) and modify SDKs.props file. If not defined
SDK at OPTIX_PATH is used.


ISSUES:
1) Assigning sampele cosine weighted hemisphere direction to ray payload cause a hang if tracing depth is higher than 2.
If payload directions is set to something simple as -ray.direction there is no hang crash (even if hemisphere direction 
is still sampled, but unused). Even using only 2x2 launch dimension and having TdrDelay set to 5 seconds.

In Optix 3.5.1 using Sbvh acceleration structure builder and Bvh traverser caused hangs whenever trace depth was higher 
than 1, even when simply setting new prd.direction for new ray as negation of incident direction -ray.direction. 
Switching to Trbvh fixed this, but failed when using proper hemisphere sampling as described before. I am no longer able
to reproduce this behaviour in OptiX 3.6.

While trying to find a solution by simplifying kernels, changing acceleration structures, updating SDKs and drivers the
following error Cuda error codes were observed - 700, 702 (most often), 716, 719, 999.


2) Using rtPrintf() within a loop causes exceptions with message "Error ir rtPrintf format string" if rtLaunchIndex variable
is not used within the loop. Optix 3.5.1 and 3.6 behave slightly differently which is noted in the comments of test_generator.cu.


3) Output form rtPrinf() doesn't show up if program output redirected to file (e.g. using "program.exe > out.log 2>&1")


DIDN'T WORK ON:
#1
HW: GeFroce GTX 770, Intel i7 4770K
SW: Win 8.1 x64 Pro, video driver 337.88,
Build: Optix 3.5.1 / Cuda 5.5, Optix 3.6 / Cuda 6 based VS2012-64bit, VS2010-32bit builds for compute_20,sm_20

#2
HW: GeFroce GT525M, Intel i7 2630QM
SW: Win 8.1 x64 Pro, video driver 337.88
Build: Optix 3.5.1 / Cuda 5.5, Optix 3.6 / Cuda 6 based VS2012-64bit, VS2010-32bit builds for compute_20,sm_20

#3
HW: GeFroce GTX 760
SW: Win 8.1 x64 Pro, video driver 335.23
Build: Optix 3.6 / Cuda 6 based VS2012-64bit, VS2010-32bit builds for compute_20,sm_20

#4
HW: Quadro FX 5800
SW: Win 8.1 x64, video driver ????
Build: Optix 3.0.1 / Cuda 4 based VS2010-32bit build for compute_11,sm_11

#5
HW: GeFroce 310
SW: Win 8.1 x64 Pro, video driver ????
Build: Optix 3.6 / Cuda 6 based VS2010-32bit builds for for compute_11,sm_11

WORKED ON:
#1
HW: GeFroce 8600M GT
SW: Win 8.1 x64 Pro, video driver 335.23
Build: Optix 3.6 based VS2012-64bit build for compute_11,sm_11

#2
HW: Quadro FX 1600M
SW: Win 7 x64 Enterprise SP1, video driver 334.95
Build: Optix 3.6 based VS2012-64bit build for compute_11,sm_11