cmake_minimum_required(VERSION 3.18)
# include(ExternalProject)

# set project name and languages
project(Ping_Pong CXX C Fortran CUDA)

set(CMAKE_CXX_FLAGS "-DMPICH_SKIP_MPICXX -arch=sm_70")
set(CMAKE_CUDA_FLAGS "-DMPICH_SKIP_MPICXX -arch=sm_70")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_CXX_COMPILER "${CMAKE_SOURCE_DIR}/kokkos/bin/nvcc_wrapper -ccbin xlc++") 

# MPI
find_package(MPI REQUIRED)
find_package(Kokkos 3 REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
#include_directories(SYSTEM ${CUDA_HOME}/include/)
set(CMAKE_CUDA_ARCHITECTURES 70)

#set(TEMPI_ENABLE_KAHIP OFF)
#set(TEMPI_OUTPUT_LEVEL SPEW)
#add_subdirectory(tempi)

##############################################################################
## Copyright (c) 2018-2020, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory
##
## LLNL-CODE-758885
##
## All rights reserved.
##
## This file is part of Comb.
##
## For details, see https://github.com/LLNL/Comb
## Please also see the LICENSE file for MIT license.
##############################################################################

find_path(GDSYNC_PATH
    NAMES "lib/libgdsync.so"
    PATHS
      ENV GDSYNC_DIR
      /opt/ibm/spectrum_mpi/libgdsync
    DOC "Path to gdsync library")


if(GDSYNC_PATH)
    message(STATUS "GDSYNC_PATH:  ${GDSYNC_PATH}")
    set(GDSYNC_FOUND TRUE)
    set(GDSYNC_CXX_COMPILE_FLAGS -I${GDSYNC_PATH}/include)
    set(GDSYNC_INCLUDE_PATH      ${GDSYNC_PATH}/include)
    set(GDSYNC_CXX_LINK_FLAGS    -L${GDSYNC_PATH}/lib)
    set(GDSYNC_CXX_LIBRARIES     ${GDSYNC_PATH}/lib/libgdsync.so)
    set(GDSYNC_ARCH              )
else()
    set(GDSYNC_FOUND FALSE)
    message(WARNING "gdsync library not found")
endif()

#
find_path(MP_PATH
    NAMES "lib/libmp.so"
    PATHS
      ENV MP_DIR
      /opt/ibm/spectrum_mpi/libmp
    DOC "Path to mp library")


if(MP_PATH)
    message(STATUS "MP_PATH:  ${MP_PATH}")
    set(MP_FOUND TRUE)
    set(MP_CXX_COMPILE_FLAGS -I${MP_PATH}/include)
    set(MP_INCLUDE_PATH      ${MP_PATH}/include)
    set(MP_CXX_LINK_FLAGS    -L${MP_PATH}/lib)
    set(MP_CXX_LIBRARIES     ${MP_PATH}/lib/libmp.so)
    set(MP_ARCH              )
else()
    set(MP_FOUND FALSE)
    message(WARNING "mp library not found")
endif()

# wrap.py setup -- grab the add_wrapped_file macro.
set(WRAP ${PROJECT_SOURCE_DIR}/wrap/wrap.py)
set(OLD_MPI_CXX_COMPILER $MPI_CXX_COMPILER)
set(MPI_CXX_COMPILER "mpicxx")
include(wrap/WrapConfig.cmake)
add_wrapped_file(nvtx_pmpi.cpp src/nvtx.w)
set(MPI_CXX_COMPILER $OLD_MPI_CXX_COMPILER)
add_library(nvtx_pmpi SHARED nvtx_pmpi.cpp)
#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/CUP-ECS/ping-pong-gpu/build-mvapich2/tempi/libtempi.so

add_executable(ping_pong src/main.cpp src/ping_pong.cpp src/input.cpp)

# Generate rule for nvtx_pmpi.o, rather than by hand
set_target_properties(ping_pong PROPERTIES CUDA_ARCHITECTURES "70")
#target_link_libraries(ping_pong PRIVATE tempi::tempi)
target_link_libraries(ping_pong PUBLIC nvtx_pmpi)
target_link_libraries(ping_pong PUBLIC Kokkos::kokkos) 
target_link_libraries(ping_pong PUBLIC MPI::MPI_CXX)
target_link_libraries(ping_pong PRIVATE nvToolsExt)
include_directories(${MP_INCLUDE_PATH})
target_link_libraries(ping_pong PRIVATE ${MP_CXX_LIBRARIES})
include_directories(${GDSYNC_INCLUDE_PATH})
target_link_libraries(ping_pong PRIVATE ${GDSYNC_CXX_LIBRARIES})

if (DIRECT)
  target_compile_definitions(DIRECT)
elseif(CUDA_AWARE)
  target_compile_definitions(CUDA_AWARE)
elseif(COPY)
  target_compile_definitions(COPY)
endif()

# set install destination
install(TARGETS ping_pong
        RUNTIME DESTINATION)

# print build strings
message("Ping Pong Build Time ${BUILD_TIME}")

#target_link_libraries(ping_pong
#                      kokkos
#                      lua)

# set kokkos architecture flags depending on detected cuda architecture
#if ("${TEST_ARCH_readable}" STREQUAL "sm_35")
#set(Kokkos_ARCH_KEPLER35 ON CACHE BOOL "")
#elseif ("${TEST_ARCH_readable}" STREQUAL "sm_60")
#set(Kokkos_ARCH_PASCAL60 ON CACHE BOOL "")
#elseif ("${TEST_ARCH_readable}" STREQUAL "sm_61")
#set(Kokkos_ARCH_PASCAL61 ON CACHE BOOL "")
#elseif ("${TEST_ARCH_readable}" STREQUAL "sm_70")
#message("TURNING ON VOLTA70")
#set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "")
#endif()

# enable other kokkos cuda options and append build type string
#set(Kokkos_ENABLE_CUDA ON CACHE BOOL "")
#set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "")
#set(Kokkos_ENABLE_CXX17 ON CACHE BOOL "") 

# Configure Kokkos
#include_directories(${Kokkos_INCLUDE_DIRS_RET})
#add_subdirectory(kokkos) 




# add built strings to source code
#if(CUDA)
#  target_compile_definitions(fiesta PRIVATE HAVE_CUDA)
#elseif(OPENMP)
#  target_compile_definitions(fiesta PRIVATE HAVE_OPENMP)
#endif()


# Switch to nvcc compiler wrapper if building for cuda
#set(CMAKE_CXX_COMPILER "nvcc_wrapper") 
#set(CMAKE_CXX_FLAGS "-ccbin xlc++")


