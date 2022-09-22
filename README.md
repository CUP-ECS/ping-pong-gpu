Basic regular mesh ping pong benchmark

Arguments:
  * -n: Length of one face of the mesh being communicated (resulting communciation is n * n * 5 * 3 doubles
  * -i: Number of iterations to perform
  * -d: Face of mesh to communicate (0 = y/z, 1 = x/z, 2 = x/y)
  * -m: Mode to use for sending and receiving (0 = MPI datatypes, 1 = Hand gpu pack, gpu-aware MPI, 2 = Hand gpu pack, host memory send)

Example command line:
``srun --mpi=pmi2 --ntasks 2 --gpus-per-task=1 --tasks-per-node=1 -p cup-ecs ping_pong -n 200 -i 100 -d 1 -m 0``

Spack configuration files for different MPIs are in the configs/ directory.
