cmake_minimum_required(VERSION 3.18)
# include(ExternalProject)

# set project name and languages
project(Ping_Pong CXX C Fortran CUDA)

# Switch to nvcc compiler wrapper if building for cuda
#set(CMAKE_CXX_COMPILER "nvcc_wrapper")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# MPI
find_package(MPI REQUIRED)
find_package(Kokkos 3 REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(SYSTEM $CUDA_HOME/include/)

add_executable(ping_pong src/main.cpp src/ping_pong.cpp src/input.cpp src/nvtx_pmpi.c)
#target_link_libraries(ping_pong
#                      kokkos
#                      lua)

#add_subdirectory(tempi)
set_target_properties(ping_pong PROPERTIES CUDA_ARCHITECTURES "70")
#target_link_libraries(ping_pong PRIVATE tempi::tempi)
target_link_libraries(ping_pong PUBLIC Kokkos::kokkos MPI::MPI_CXX)
target_link_libraries(ping_pong PRIVATE nvToolsExt)

# wrap.py setup -- grab the add_wrapped_file macro.
set(WRAP ${PROJECT_SOURCE_DIR}/wrap/wrap.py)
include(wrap/WrapConfig.cmake)

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

