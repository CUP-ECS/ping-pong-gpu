# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/tce/packages/cmake/cmake-3.20.2/bin/cmake

# The command to remove a file.
RM = /usr/tce/packages/cmake/cmake-3.20.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /g/g15/haskins8/CUP-ECS/ping-pong-gpu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /g/g15/haskins8/CUP-ECS/ping-pong-gpu/build-xlc

# Include any dependencies generated for this target.
include CMakeFiles/ping_pong.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ping_pong.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ping_pong.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ping_pong.dir/flags.make

CMakeFiles/ping_pong.dir/src/main.cpp.o: CMakeFiles/ping_pong.dir/flags.make
CMakeFiles/ping_pong.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/g/g15/haskins8/CUP-ECS/ping-pong-gpu/build-xlc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ping_pong.dir/src/main.cpp.o"
	/g/g15/haskins8/spack/var/spack/environments/spack-xlc-mvapich/.spack-env/view/bin/nvcc_wrapper  -ccbin xlc++ -x c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ping_pong.dir/src/main.cpp.o -c /g/g15/haskins8/CUP-ECS/ping-pong-gpu/src/main.cpp

CMakeFiles/ping_pong.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ping_pong.dir/src/main.cpp.i"
	/g/g15/haskins8/spack/var/spack/environments/spack-xlc-mvapich/.spack-env/view/bin/nvcc_wrapper  -ccbin xlc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /g/g15/haskins8/CUP-ECS/ping-pong-gpu/src/main.cpp > CMakeFiles/ping_pong.dir/src/main.cpp.i

CMakeFiles/ping_pong.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ping_pong.dir/src/main.cpp.s"
	/g/g15/haskins8/spack/var/spack/environments/spack-xlc-mvapich/.spack-env/view/bin/nvcc_wrapper  -ccbin xlc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /g/g15/haskins8/CUP-ECS/ping-pong-gpu/src/main.cpp -o CMakeFiles/ping_pong.dir/src/main.cpp.s

CMakeFiles/ping_pong.dir/src/ping_pong.cpp.o: CMakeFiles/ping_pong.dir/flags.make
CMakeFiles/ping_pong.dir/src/ping_pong.cpp.o: ../src/ping_pong.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/g/g15/haskins8/CUP-ECS/ping-pong-gpu/build-xlc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ping_pong.dir/src/ping_pong.cpp.o"
	/g/g15/haskins8/spack/var/spack/environments/spack-xlc-mvapich/.spack-env/view/bin/nvcc_wrapper  -ccbin xlc++ -x c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ping_pong.dir/src/ping_pong.cpp.o -c /g/g15/haskins8/CUP-ECS/ping-pong-gpu/src/ping_pong.cpp

CMakeFiles/ping_pong.dir/src/ping_pong.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ping_pong.dir/src/ping_pong.cpp.i"
	/g/g15/haskins8/spack/var/spack/environments/spack-xlc-mvapich/.spack-env/view/bin/nvcc_wrapper  -ccbin xlc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /g/g15/haskins8/CUP-ECS/ping-pong-gpu/src/ping_pong.cpp > CMakeFiles/ping_pong.dir/src/ping_pong.cpp.i

CMakeFiles/ping_pong.dir/src/ping_pong.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ping_pong.dir/src/ping_pong.cpp.s"
	/g/g15/haskins8/spack/var/spack/environments/spack-xlc-mvapich/.spack-env/view/bin/nvcc_wrapper  -ccbin xlc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /g/g15/haskins8/CUP-ECS/ping-pong-gpu/src/ping_pong.cpp -o CMakeFiles/ping_pong.dir/src/ping_pong.cpp.s

CMakeFiles/ping_pong.dir/src/input.cpp.o: CMakeFiles/ping_pong.dir/flags.make
CMakeFiles/ping_pong.dir/src/input.cpp.o: ../src/input.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/g/g15/haskins8/CUP-ECS/ping-pong-gpu/build-xlc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/ping_pong.dir/src/input.cpp.o"
	/g/g15/haskins8/spack/var/spack/environments/spack-xlc-mvapich/.spack-env/view/bin/nvcc_wrapper  -ccbin xlc++ -x c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ping_pong.dir/src/input.cpp.o -c /g/g15/haskins8/CUP-ECS/ping-pong-gpu/src/input.cpp

CMakeFiles/ping_pong.dir/src/input.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ping_pong.dir/src/input.cpp.i"
	/g/g15/haskins8/spack/var/spack/environments/spack-xlc-mvapich/.spack-env/view/bin/nvcc_wrapper  -ccbin xlc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /g/g15/haskins8/CUP-ECS/ping-pong-gpu/src/input.cpp > CMakeFiles/ping_pong.dir/src/input.cpp.i

CMakeFiles/ping_pong.dir/src/input.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ping_pong.dir/src/input.cpp.s"
	/g/g15/haskins8/spack/var/spack/environments/spack-xlc-mvapich/.spack-env/view/bin/nvcc_wrapper  -ccbin xlc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /g/g15/haskins8/CUP-ECS/ping-pong-gpu/src/input.cpp -o CMakeFiles/ping_pong.dir/src/input.cpp.s

# Object files for target ping_pong
ping_pong_OBJECTS = \
"CMakeFiles/ping_pong.dir/src/main.cpp.o" \
"CMakeFiles/ping_pong.dir/src/ping_pong.cpp.o" \
"CMakeFiles/ping_pong.dir/src/input.cpp.o"

# External object files for target ping_pong
ping_pong_EXTERNAL_OBJECTS =

ping_pong: CMakeFiles/ping_pong.dir/src/main.cpp.o
ping_pong: CMakeFiles/ping_pong.dir/src/ping_pong.cpp.o
ping_pong: CMakeFiles/ping_pong.dir/src/input.cpp.o
ping_pong: CMakeFiles/ping_pong.dir/build.make
ping_pong: ../nvtx_pmpi.o
ping_pong: /g/g15/haskins8/spack/var/spack/environments/spack-xlc-mvapich/.spack-env/view/lib64/libkokkoscontainers.so.3.2.0
ping_pong: /g/g15/haskins8/spack/var/spack/environments/spack-xlc-mvapich/.spack-env/view/lib64/libkokkoscore.so.3.2.0
ping_pong: /usr/lib64/libdl.so
ping_pong: /usr/tce/packages/cuda/cuda-11.1.1/lib64/libcuda.so
ping_pong: /usr/tce/packages/cuda/cuda-11.1.1/lib64/libcudart.so
ping_pong: /lib64/librt.so
ping_pong: /usr/tce/packages/spectrum-mpi/ibm/spectrum-mpi-2020.08.19/lib/libpmix.so
ping_pong: /lib64/libm.so
ping_pong: /usr/tce/packages/mvapich2/mvapich2-2020.12.11-cuda-10.1.243-xl-2021.03.11/lib64/libmpicxx.so
ping_pong: /usr/tce/packages/mvapich2/mvapich2-2020.12.11-cuda-10.1.243-xl-2021.03.11/lib64/libmpi.so
ping_pong: CMakeFiles/ping_pong.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/g/g15/haskins8/CUP-ECS/ping-pong-gpu/build-xlc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ping_pong"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ping_pong.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ping_pong.dir/build: ping_pong
.PHONY : CMakeFiles/ping_pong.dir/build

CMakeFiles/ping_pong.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ping_pong.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ping_pong.dir/clean

CMakeFiles/ping_pong.dir/depend:
	cd /g/g15/haskins8/CUP-ECS/ping-pong-gpu/build-xlc && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /g/g15/haskins8/CUP-ECS/ping-pong-gpu /g/g15/haskins8/CUP-ECS/ping-pong-gpu /g/g15/haskins8/CUP-ECS/ping-pong-gpu/build-xlc /g/g15/haskins8/CUP-ECS/ping-pong-gpu/build-xlc /g/g15/haskins8/CUP-ECS/ping-pong-gpu/build-xlc/CMakeFiles/ping_pong.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ping_pong.dir/depend

