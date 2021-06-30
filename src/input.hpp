#include <stdio.h>
#include "mpi.h"
#include "Kokkos_Core.hpp"
#include "KokkosTypes.hpp"

//#ifndef DIRECT
//#define DIRECT
//#endif
//
//#ifndef CUDA_AWARE
//#define CUDA_AWARE
//#endif
//
//#ifndef COPY
//#define COPY
//#endif

struct inputConfig {
  int nci, ncj, nck;
  int ng, ngi, ngj, ngk, nvt;
};

struct inputConfig executeConfiguration(int max_i);
