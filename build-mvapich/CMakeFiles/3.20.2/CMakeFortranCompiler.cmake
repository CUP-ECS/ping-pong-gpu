set(CMAKE_Fortran_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2021.05.28-cuda-11.1.1-gcc-8.3.1/bin/mpifort")
set(CMAKE_Fortran_COMPILER_ARG1 "")
set(CMAKE_Fortran_COMPILER_ID "GNU")
set(CMAKE_Fortran_COMPILER_VERSION "8.3.1")
set(CMAKE_Fortran_COMPILER_WRAPPER "")
set(CMAKE_Fortran_PLATFORM_ID "Linux")
set(CMAKE_Fortran_SIMULATE_ID "")
set(CMAKE_Fortran_SIMULATE_VERSION "")




set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_Fortran_COMPILER_AR "/usr/bin/gcc-ar")
set(CMAKE_RANLIB "/usr/bin/ranlib")
set(CMAKE_Fortran_COMPILER_RANLIB "/usr/bin/gcc-ranlib")
set(CMAKE_COMPILER_IS_GNUG77 1)
set(CMAKE_Fortran_COMPILER_LOADED 1)
set(CMAKE_Fortran_COMPILER_WORKS TRUE)
set(CMAKE_Fortran_ABI_COMPILED TRUE)
set(CMAKE_COMPILER_IS_MINGW )
set(CMAKE_COMPILER_IS_CYGWIN )
if(CMAKE_COMPILER_IS_CYGWIN)
  set(CYGWIN 1)
  set(UNIX 1)
endif()

set(CMAKE_Fortran_COMPILER_ENV_VAR "FC")

set(CMAKE_Fortran_COMPILER_SUPPORTS_F90 1)

if(CMAKE_COMPILER_IS_MINGW)
  set(MINGW 1)
endif()
set(CMAKE_Fortran_COMPILER_ID_RUN 1)
set(CMAKE_Fortran_SOURCE_FILE_EXTENSIONS f;F;fpp;FPP;f77;F77;f90;F90;for;For;FOR;f95;F95)
set(CMAKE_Fortran_IGNORE_EXTENSIONS h;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_Fortran_LINKER_PREFERENCE 20)
if(UNIX)
  set(CMAKE_Fortran_OUTPUT_EXTENSION .o)
else()
  set(CMAKE_Fortran_OUTPUT_EXTENSION .obj)
endif()

# Save compiler ABI information.
set(CMAKE_Fortran_SIZEOF_DATA_PTR "8")
set(CMAKE_Fortran_COMPILER_ABI "")
set(CMAKE_Fortran_LIBRARY_ARCHITECTURE "")

if(CMAKE_Fortran_SIZEOF_DATA_PTR AND NOT CMAKE_SIZEOF_VOID_P)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_Fortran_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_Fortran_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_Fortran_COMPILER_ABI}")
endif()

if(CMAKE_Fortran_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()





set(CMAKE_Fortran_IMPLICIT_INCLUDE_DIRECTORIES "/usr/tce/packages/mvapich2/osu/mvapich2-2021.05.28-cuda-11.1.1/gcc-8.3.1/include;/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib/gcc/ppc64le-redhat-linux/8/finclude;/usr/tce/packages/cuda/cuda-11.1.1/include;/g/g15/haskins8/spack/var/spack/environments/mvapich-gcc8/.spack-env/view/include;/usr/tce/packages/mvapich2/mvapich2-2020.12.11-cuda-10.1.243-gcc-8.3.1/include;/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib/gcc/ppc64le-redhat-linux/8/include;/usr/local/include;/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/include;/usr/include")
set(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES "cuda;cudart;rt;pmix;gfortran;m;cupti;mpifort;mpi;gfortran;m;gcc_s;gcc;quadmath;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES "/usr/tce/packages/cuda/cuda-11.1.1/lib64/stubs;/usr/tce/packages/cuda/cuda-11.1.1/lib64;/usr/tce/packages/spectrum-mpi/ibm/spectrum-mpi-rolling-release/lib;/usr/tce/packages/cuda/cuda-10.1.243/extras/CUPTI/lib64;/usr/tce/packages/mvapich2/osu/mvapich2-2021.05.28-cuda-11.1.1/gcc-8.3.1/lib64;/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib/gcc/ppc64le-redhat-linux/8;/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib/gcc;/g/g15/haskins8/spack/var/spack/environments/mvapich-gcc8/.spack-env/view/lib64;/usr/tce/packages/mvapich2/mvapich2-2020.12.11-cuda-10.1.243-gcc-8.3.1/lib64;/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib64;/lib64;/usr/lib64;/usr/tce/packages/gcc/gcc-8.3.1/rh/usr/lib")
set(CMAKE_Fortran_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
