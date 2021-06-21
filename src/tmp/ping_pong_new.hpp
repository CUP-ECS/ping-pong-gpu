#include "Kokkos_Core.hpp"
#include "KokkosTypes.hpp"
#include "input.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

//#define FS_LAYOUT Kokkos::DefaultExecutionSpace::array_layout
#define MPI_TAG2 4

//typedef typename Kokkos::View<double ****, FS_LAYOUT> FS4D;
//typedef typename Kokkos::View<double ****, FS_LAYOUT>::HostMirror FS4DH;
//typedef typename Kokkos::View<double *, FS_LAYOUT> FS1D;

//MPI_Datatype leftRecvSubArray, rightRecvSubArray;
//MPI_Datatype leftSendSubArray, rightSendSubArray;
//
//FS4D leftSend, leftRecv;
//FS4D rightSend, rightRecv;
//
//FS4DH leftSend_H, leftRecv_H;
//FS4DH rightSend_H, rightRecv_H;

void direct(int rank, int n_iterations, FS4D a, FS1D aR, FS1D aS, inputConfig cf, int mode);

void cuda_aware(int rank, int n_iterations, FS4D a, FS1D aR, FS1D aS, inputConfig cf, int mode);
void copy(int rank, int n_iterations, FS4D a, FS1D aR, FS1D aS, inputConfig cf, int mode);

void send_recv(int rank, int n_iterations, FS4D a, FS1D aR, FS1D aS, inputConfig cf, int mode);

void ping_pong_n_dim(int max_i, int n_iterations, int dimension, int mode);
