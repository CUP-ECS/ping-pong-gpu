#include "Kokkos_Core.hpp"
#include "KokkosTypes.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

//#define FS_LAYOUT Kokkos::DefaultExecutionSpace::array_layout
#define MPI_TAG2 4

//typedef typename Kokkos::View<double ****, FS_LAYOUT> FS4D;
//typedef typename Kokkos::View<double ****, FS_LAYOUT>::HostMirror FS4DH;
//typedef typename Kokkos::View<double *, FS_LAYOUT> FS1D;

MPI_Datatype leftRecvSubArray, rightRecvSubArray;
MPI_Datatype leftSendSubArray, rightSendSubArray;

FS4D leftSend, leftRecv;
FS4D rightSend, rightRecv;

FS4DH leftSend_H, leftRecv_H;
FS4DH rightSend_H, rightRecv_H;

void ping_pong_n_dim( int max_i, int n_iterations, int dimension, int mode );
