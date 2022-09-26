#!/bin/bash
#SBATCH --job-name=BeatnikTest # Job name
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00              # Time limit hrs:min:sec
#SBATCH --output=PingPongGPU.%j.log # Standard output and error log
#SBATCH --partition=cup-ecs

SPACK_ENVIRONMENT_HOME=/users/bridges/workspace/ping-pong-gpu/build-hopper
PINGPONG_BINARY_HOME=/users/bridges/workspace/ping-pong-gpu/build-hopper-openmpi
MESH_SIZE="100 200 400 600 800 1000"
echo "Loading compiler"
module load gcc/11.2.0

echo "Loading spack development environment"
source /users/bridges/spack/share/spack/setup-env.sh
spack env deactivate
spack env activate -d ${SPACK_ENVIRONMENT_HOME}

echo "Changing to ${PINGPONG_BINARY_HOME} to run"
cd ${PINGPONG_BINARY_HOME}

#for d in 0 1 2
for d in 2
do 
  echo "Testing ping pong direction=$d"
  # for m in 0 1 2
  for m in 2
  do
  echo "Testing ping pong mode=$m"
  # for n in $MESH_SIZE
  for n in 1000
    do
    echo "Testing ping pong size=$n"
    srun --mpi=pmi2 -n 2 ${PINGPONG_BINARY_HOME}/ping_pong -d $d -n $n -i 500 -m $m
    done
  done
done
echo "Finished MPI Run"
