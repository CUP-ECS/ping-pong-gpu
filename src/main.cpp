#include "KokkosTypes.hpp"
#include "ping_pong.hpp"
//#include "input.hpp"
#include <stdio.h>
#include <getopt.h>
#include <iostream>
#include <mpi.h>
#include <chrono>

using namespace std;

static char* shortargs = (char*)"n:d:b:i:m:";

static option longargs[] = {
    // Basic simulation parameters
    { "size", required_argument, NULL, 'n' },
    { "direction", required_argument, NULL, 'd' },
    { "buffers", required_argument, NULL, 'b' },
    { "iterations", required_argument, NULL, 'i' },
    { "mode", required_argument, NULL, 'm' }
};

int main( int argc, char *argv[] ) {

  int rank, num_procs;
  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &num_procs );
  Kokkos::initialize( argc, argv );

  int max_i        = 20;
  int n_iterations = 1000;
  int buffers      = 1;
  int mode         = 0;
  int direction    = 0;

    int ch;
    // Now parse any arguments
    while ( ( ch = getopt_long( argc, argv, shortargs, longargs, NULL ) ) !=
            -1 )
    {
        switch ( ch )
        {
        case 'n':
            max_i = atoi( optarg );
            break;
        case 'i':
            n_iterations = atoi( optarg );
            break;
        case 'd':
            direction = atoi( optarg );
            break;
        case 'm':
            mode = atoi( optarg );
            break;
        case 'b':
            buffers = atoi( optarg );
            break;
        }
    }

    ping_pong_n_dim( max_i, n_iterations, buffers, mode, direction );

    MPI_Finalize();
    Kokkos::finalize();
    return 0;
}
