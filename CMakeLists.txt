cmake_minimum_required(VERSION 3.10)
# include(ExternalProject)

# Switch to nvcc compiler wrapper if building for cuda
if (CUDA)
  set(CMAKE_CXX_COMPILER "${CMAKE_SOURCE_DIR}/kokkos/bin/nvcc_wrapper")
elseif(AMDGPU)
  set(CMAKE_CXX_COMPILER "hipcc")
endif()

# set project name and languages
project(Ping_Pong CXX C Fortran)

set(CMAKE_CXX_STANDARD 17)

# set kokkos architecture flags depending on detected cuda architecture
if ("${TEST_ARCH_readable}" STREQUAL "sm_35")
  set(Kokkos_ARCH_KEPLER35 ON CACHE BOOL "")
elseif ("${TEST_ARCH_readable}" STREQUAL "sm_60")
  set(Kokkos_ARCH_PASCAL60 ON CACHE BOOL "")
elseif ("${TEST_ARCH_readable}" STREQUAL "sm_61")
  set(Kokkos_ARCH_PASCAL61 ON CACHE BOOL "")
elseif ("${TEST_ARCH_readable}" STREQUAL "sm_70")
  message("TURNING ON VOLTA70")
  set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "")
endif()

  # enable other kokkos cuda options and append build type string
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "")
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "")
  set(Kokkos_ENABLE_CXX17 ON CACHE BOOL "")
  set(FIESTA_OPTS "${FIESTA_OPTS}+CUDA")
elseif(AMDGPU)
  set(Kokkos_ENABLE_HIP ON CACHE BOOL "")
  set(Kokkos_ARCH_VEGA900 ON CACHE BOOL "")
  set(FIESTA_OPTS "${FIESTA_OPTS}+AMD")
elseif(OPENMP)
  # set kokkos openmp optins and append build type string
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "")
  set(FIESTA_OPTS "${FIESTA_OPTS}+OPENMP")
endif()

# Configure Kokkos
include_directories(${Kokkos_INCLUDE_DIRS_RET})
add_subdirectory(kokkos)

# Configure Lua
include_directories(lua/src)
add_subdirectory(lua)

# MPI
find_package(MPI REQUIRED)
find_package(Kokkos 3 REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})


# include third party headers (might be empty if third party components are provided elsewhere)
include_directories(${installDir}/include)

# set base source files
set(BASE_SRCS src/ping_pong.cpp)

add_executable(ping_pong ${FIESTA_SRCS})
target_link_libraries(ping_pong
                      kokkos
                      lua)

if (TEMPI)
  target_link_libraries(ping_pong PRIVATE tempi)
endif (TEMPI)

# add built strings to source code
if(CUDA)
  target_compile_definitions(fiesta PRIVATE HAVE_CUDA)
elseif(OPENMP)
  target_compile_definitions(fiesta PRIVATE HAVE_OPENMP)
endif()

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
