BUILDING:
1. Define OPTIX_PATH environment variable pointing to OptiX SDK. 
2. Define CUDA_USE_VER environment variable as "5.5" or "6.0". It controls which Cuda 
build customization .props and .targets files will be imported into projects.
3. Build ContextTest.sln

Possible to use also OPTIX_USE_VER to quickly switch between different OptiX SDKs, but then 
need to define their path variables (e.g. OPTIX_PATH_V3_5_1 similarly to Cuda vars) and 
modify SDKs.props file. If not defined SDK at OPTIX_PATH is used.
