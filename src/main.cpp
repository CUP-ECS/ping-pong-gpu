#include "kokkosTypes.hpp"
#include "n_dimension.hpp"
#include "ping_pong.hpp"
#include "input.hpp"
#include <stdio.h>
#include <iostream>
#include <mpi.h>

using namespace std;

int main( int argc, char *argv[] ) {

  struct inputConfig cf = executeConfiguration();
  int rank, num_procs;
  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &num_procs );
  Kokkos::initialize( argc, argv );

  int max_i        = 20;
  int n_iterations = 1000;
  int dimension    = 4;

  if ( argc > 1 ) max_i        = atoi( argv[1] );
  if ( argc > 2 ) n_iterations = atoi( argv[2] );
  //if ( argc > 3 ) dimension    = atoi( argv[3] );

  ping_pong_n_dim( cf, max_i, n_iterations, dimension );

  MPI_Finalize();

  return 0;
}
