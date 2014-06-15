This project is made reproduce OptiX hangs/crashes dicussed in this Optix forum thread
https://devtalk.nvidia.com/default/topic/751906/optix/weird-ray-generation-hang-really-simple-code-/

To compile it requires CUDA_PATH and OPTIX_PATH environment variables defined pointing to 
respective installation directories.

There are two issues:
1) Assigning sampeled cosine weighted hemisphere direction to ray payload cause a hang if tracing depth is higher than 2.
If payload directions is set to something simple as -ray.direction there is no hang crash (even if hemisphere direction 
is still sampled, but unused). Even using only 2x2 launch dimension and having TdrDelay set to 5 seconds.

In Optix 3.5.1 using Sbvh acceleration structure builder and Bvh traverser caused hangs whenever trace depth was higher 
than 1, even when simply setting new prd.direction for new ray as negation of incident direction -ray.direction. 
Switching to Trbvh wixed this, but failed when using proper hemisphere sampling as described before. I am no longer able
to reproduce this behavior in OptiX 3.6.

2) Using rtPrintf() within a loop causes exceptions with message "Error ir rtPrintf format string" if rtLaunchIndex variable
is not used within the loop. Optix 3.5.1 and 3.6 behave slightly differently wich is noted in the comments in the 
test_generator.cu.

3) Output form rtPrinf() doesn't show up if program output redirected to file (e.g. using "program.exe > out.log 2>&1")

All issues were reproduced on two Window 8.1 x64 machines:
#1
HW: GeFroce GTX 770, Intel i7 4770K
SW: Optix 3.5.1 / Cuda 5.5, Optix 3.6 / Cuda 6, video driver 337.88, VS2012 (64bit builds), VS2010 (32bit builds)

#2
HW: GeFroce GT525M, Intel i7 2630QM
SW: Optix 3.5.1 / Cuda 5.5, Optix 3.6 / Cuda 6, video driver 337.88, VS2012 (64bit builds), VS2010 (32bit builds)
ssd