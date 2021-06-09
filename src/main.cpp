#include "KokkosTypes.hpp"
#include "ping_pong.hpp"
//#include "input.hpp"
#include <stdio.h>
#include <iostream>
#include <mpi.h>
#include <chrono>

using namespace std;

int main( int argc, char *argv[] ) {

  int rank, num_procs;
  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &num_procs );
  Kokkos::initialize( argc, argv );

  int max_i        = 20;
  int n_iterations = 1000;
  int dimension    = 4;
  int mode         = 0;

  if ( argc > 1 ) max_i        = atoi( argv[1] );
  if ( argc > 2 ) n_iterations = atoi( argv[2] );
  if ( argc > 3 ) dimension    = atoi( argv[3] );
  if ( argc > 4 ) mode         = atoi( argv[4] );

  ping_pong_n_dim( max_i, n_iterations, dimension, mode );

  MPI_Finalize();

  Kokkos::finalize();
  return 0;
}
