set(CMAKE_C_COMPILER "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-xl-2021.03.11/bin/mpicc")
set(CMAKE_C_COMPILER_ARG1 "")
set(CMAKE_C_COMPILER_ID "XLClang")
set(CMAKE_C_COMPILER_VERSION "16.1.1.10")
set(CMAKE_C_COMPILER_VERSION_INTERNAL "")
set(CMAKE_C_COMPILER_WRAPPER "")
set(CMAKE_C_STANDARD_COMPUTED_DEFAULT "99")
set(CMAKE_C_COMPILE_FEATURES "c_std_90;c_function_prototypes;c_std_99;c_restrict;c_variadic_macros;c_std_11;c_static_assert")
set(CMAKE_C90_COMPILE_FEATURES "c_std_90;c_function_prototypes")
set(CMAKE_C99_COMPILE_FEATURES "c_std_99;c_restrict;c_variadic_macros")
set(CMAKE_C11_COMPILE_FEATURES "c_std_11;c_static_assert")

set(CMAKE_C_PLATFORM_ID "Linux")
set(CMAKE_C_SIMULATE_ID "")
set(CMAKE_C_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_C_SIMULATE_VERSION "")




set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_C_COMPILER_AR "")
set(CMAKE_RANLIB "/usr/bin/ranlib")
set(CMAKE_C_COMPILER_RANLIB "")
set(CMAKE_LINKER "/usr/tcetmp/bin/ld")
set(CMAKE_MT "")
set(CMAKE_COMPILER_IS_GNUCC )
set(CMAKE_C_COMPILER_LOADED 1)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_C_ABI_COMPILED TRUE)
set(CMAKE_COMPILER_IS_MINGW )
set(CMAKE_COMPILER_IS_CYGWIN )
if(CMAKE_COMPILER_IS_CYGWIN)
  set(CYGWIN 1)
  set(UNIX 1)
endif()

set(CMAKE_C_COMPILER_ENV_VAR "CC")

if(CMAKE_COMPILER_IS_MINGW)
  set(MINGW 1)
endif()
set(CMAKE_C_COMPILER_ID_RUN 1)
set(CMAKE_C_SOURCE_FILE_EXTENSIONS c;m)
set(CMAKE_C_IGNORE_EXTENSIONS h;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_C_LINKER_PREFERENCE 10)

# Save compiler ABI information.
set(CMAKE_C_SIZEOF_DATA_PTR "8")
set(CMAKE_C_COMPILER_ABI "ELF")
set(CMAKE_C_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_C_LIBRARY_ARCHITECTURE "")

if(CMAKE_C_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_C_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_C_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_C_COMPILER_ABI}")
endif()

if(CMAKE_C_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_C_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_C_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES "/usr/tce/packages/xl/xl-2021.03.11/xlsmp/5.1.1/include;/usr/tce/packages/xl/xl-2021.03.11/xlmass/9.1.1/include;/usr/tce/packages/xl/xl-2021.03.11/xlC/16.1.1/include;/usr/local/include;/usr/tce/packages/gcc/gcc-4.9.3/gnu/include;/usr/include;/usr/tce/packages/spectrum-mpi/ibm/spectrum-mpi-rolling-release/include")
set(CMAKE_C_IMPLICIT_LINK_LIBRARIES "mpiprofilesupport;mpi_ibm;xlopt;xl;dl;gcc_s;pthread;gcc;m;c;gcc_s;gcc")
set(CMAKE_C_IMPLICIT_LINK_DIRECTORIES "/usr/tce/packages/spectrum-mpi/ibm/spectrum-mpi-rolling-release/lib;/g/g15/haskins8/spack/var/spack/environments/spack-xlc-spectrum/.spack-env/view/lib64;/usr/tce/packages/cuda/cuda-11.1.1/lib64;/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-xl-2021.03.11/lib;/g/g15/haskins8/spack/var/spack/environments/spack-xlc-spectrum/.spack-env/view/lib;/usr/tce/packages/xl/xl-2021.03.11/xlsmp/5.1.1/lib;/usr/tce/packages/xl/xl-2021.03.11/xlmass/9.1.1/lib;/usr/tce/packages/xl/xl-2021.03.11/xlC/16.1.1/lib;/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3;/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64;/lib64;/usr/lib64")
set(CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
