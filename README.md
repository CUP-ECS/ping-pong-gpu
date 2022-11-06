# GPU Ping Pong Benchmark

Basic regular mesh ping pong benchmark for GPUs written in Kokkos. Baseline for 
comparison is a Kokkos parallel loop that packs the data prior to sending. Mesh
data structure extracted from UNM Fiesta CFD application.

## Running

Arguments:
  * -n: Length of one face of the mesh being communicated (resulting communciation is n * n * 5 * 3 doubles
  * -i: Number of iterations to perform
  * -d: Face of mesh to communicate (0 = y/z, 1 = x/z, 2 = x/y)
  * -m: Mode to use for sending and receiving (0 = MPI datatypes, 1 = Hand gpu pack, gpu-aware MPI, 2 = Hand gpu pack, host memory send)

Example command line:
``srun --mpi=pmi2 --ntasks 2 --gpus-per-task=1 --tasks-per-node=1 -p cup-ecs ping_pong -n 200 -i 100 -d 1 -m 0``

## Building

Spack configuration files for different MPIs are in the configs/ directory. Generally 
we create spack environments for building, use a setup script to load any necessary
modules (generally the compiler, which the spack environment doesn't necessarily
provide), and activate the spack environment for the buiuld

## Future features

  1. Option to pack into pinned host memory instead of GPU memory
  1. Restructure to support other ping pong of data structures extracted from
     other applications.
  1. Option to change number of ghost cell layers sent
