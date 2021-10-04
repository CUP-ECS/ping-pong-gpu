cmake_minimum_required(VERSION 3.18)
# include(ExternalProject)

# set project name and languages
project(Ping_Pong CXX C Fortran CUDA)

# Switch to nvcc compiler wrapper if building for cuda
#set(CMAKE_CXX_COMPILER "nvcc_wrapper") 
#set(CMAKE_CXX_FLAGS "-ccbin xlc++")



set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_CXX_COMPILER "${CMAKE_SOURCE_DIR}/kokkos/bin/nvcc_wrapper -ccbin xlc++") 

# MPI
find_package(MPI REQUIRED)
find_package(Kokkos 3 REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(SYSTEM $CUDA_HOME/include/)

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

#add_subdirectory(tempi)
# wrap.py setup -- grab the add_wrapped_file macro.
set(WRAP ${PROJECT_SOURCE_DIR}/wrap/wrap.py)
set(OLD_MPI_CXX_COMPILER $MPI_CXX_COMPILER)
set(MPI_CXX_COMPILER "mpicxx")
include(wrap/WrapConfig.cmake)
add_wrapped_file(nvtx_pmpi.cpp src/nvtx.w -f)
set(MPI_CXX_COMPILER $OLD_MPI_CXX_COMPILER)

add_executable(ping_pong src/main.cpp src/ping_pong.cpp src/input.cpp nvtx_pmpi.cpp)

# Generate rule for nvtx_pmpi.o, rather than by hand
set_target_properties(ping_pong PROPERTIES CUDA_ARCHITECTURES "70")
#target_link_libraries(ping_pong PUBLIC ${PROJECT_SOURCE_DIR}/nvtx_pmpi.o)
#target_link_libraries(ping_pong PRIVATE tempi::tempi)
target_link_libraries(ping_pong PUBLIC Kokkos::kokkos) 
target_link_libraries(ping_pong PUBLIC MPI::MPI_CXX)
target_link_libraries(ping_pong PRIVATE nvToolsExt)



# add built strings to source code
#if(CUDA)
#  target_compile_definitions(fiesta PRIVATE HAVE_CUDA)
#elseif(OPENMP)
#  target_compile_definitions(fiesta PRIVATE HAVE_OPENMP)
#endif()

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

