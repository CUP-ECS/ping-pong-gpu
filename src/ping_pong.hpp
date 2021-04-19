#include "Kokkos_Core.hpp"
#include "KokkosTypes.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void ping_pong_n_dim( int max_i, int n_iterations, int dimension );

void mpi_init(struct inputConfig &cf);

class ping_pong {

public:
  mpiBuffers(struct inputConfig cf);

#ifndef DIRECT
  FS4D a;
  FS4D b;

  FS4D send;
#endif

};

void haloExchange(struct inputConfig cf, FS4D &deviceV, class mpiBuffers &m);
