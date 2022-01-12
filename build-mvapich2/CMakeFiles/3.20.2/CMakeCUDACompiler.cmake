set(CMAKE_CUDA_COMPILER "/usr/tce/packages/cuda/cuda-11.1.1/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/usr/tce/packages/gcc/gcc-8.3.1/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "11.1.105")
set(CMAKE_CUDA_DEVICE_LINKER "/usr/tce/packages/cuda/cuda-11.1.1/nvidia/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/usr/tce/packages/cuda/cuda-11.1.1/nvidia/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "14")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "8.3")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/usr/tce/packages/cuda/cuda-11.1.1/nvidia")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/usr/tce/packages/cuda/cuda-11.1.1/nvidia")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/usr/tce/packages/cuda/cuda-11.1.1/nvidia")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/usr/tce/packages/cuda/cuda-11.1.1/nvidia/targets/ppc64le-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/usr/tce/packages/cuda/cuda-11.1.1/nvidia/targets/ppc64le-linux/lib/stubs;/usr/tce/packages/cuda/cuda-11.1.1/nvidia/targets/ppc64le-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/collab/usr/global/tools/hpctoolkit/blueos_3_ppc64le_ib/.install/2020-08/packages/linux-rhel7-ppc64le/gcc-7.3.1/hpctoolkit-2020.08.03-owvnf3dby5c6r7lavcmajluzbinwzax2/include;/usr/tce/packages/cuda/cuda-11.1.1/include;/g/g15/haskins8/spack/var/spack/environments/mvapich-gcc8/.spack-env/view/include;/usr/tce/packages/mvapich2/mvapich2-2021.05.28-cuda-11.1.1-gcc-8.3.1/include;/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/include/c++/8;/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/include/c++/8/ppc64le-redhat-linux;/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/include/c++/8/backward;/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib/gcc/ppc64le-redhat-linux/8/include;/usr/local/include;/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/include;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/usr/tce/packages/cuda/cuda-11.1.1/nvidia/targets/ppc64le-linux/lib/stubs;/usr/tce/packages/cuda/cuda-11.1.1/nvidia/targets/ppc64le-linux/lib;/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib/gcc/ppc64le-redhat-linux/8;/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib/gcc;/g/g15/haskins8/spack/var/spack/environments/mvapich-gcc8/.spack-env/view/lib64;/usr/tce/packages/mvapich2/mvapich2-2021.05.28-cuda-11.1.1-gcc-8.3.1/lib64;/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib64;/lib64;/usr/lib64;/collab/usr/global/tools/hpctoolkit/blueos_3_ppc64le_ib/.install/2020-08/packages/linux-rhel7-ppc64le/gcc-7.3.1/hpctoolkit-2020.08.03-owvnf3dby5c6r7lavcmajluzbinwzax2/lib;/usr/tce/packages/cuda/cuda-11.1.1/lib64;/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/tcetmp/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")