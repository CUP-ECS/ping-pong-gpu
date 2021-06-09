#include "ping_pong.hpp"
#include "input.hpp"
#include <iostream>
#include "Kokkos_Core.hpp"
#include "KokkosTypes.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <chrono>

using namespace std;

//#define DIRECT
//#define CUDA_AWARE
//#define COPY

//MPI_Datatype leftRecvSubArray, rightRecvSubArray;
//MPI_Datatype leftSendSubArray, rightSendSubArray;
//
//FS4D leftSend, leftRecv;
//FS4D rightSend, rightRecv;
//
//FS4DH leftSend_H, leftRecv_H;
//FS4DH rightSend_H, rightRecv_H;

void send_recv( int rank, int n_iterations,    FS4D a, 
                FS1D aR,  FS1D aS, inputConfig cf, int mode ) {

  auto xPol = Kokkos::MDRangePolicy<Kokkos::Rank<4>>( {0, 0, 0, 0},
                                                      {cf.ng, cf.ngj, cf.ngk, cf.nvt} );

  //leftSend  = Kokkos::View<double****,FS_LAYOUT>("leftSend",
  //                                               cf.ng,cf.ngj,cf.ngk,cf.nvt);
  //leftRecv  = Kokkos::View<double****,FS_LAYOUT>("leftRecv",
  //                                               cf.ng,cf.ngj,cf.ngk,cf.nvt);
  //rightSend = Kokkos::View<double****,FS_LAYOUT>("rightSend",
  //                                               cf.ng,cf.ngj,cf.ngk,cf.nvt);
  //rightRecv = Kokkos::View<double****,FS_LAYOUT>("rightRecv",
  //                                               cf.ng,cf.ngj,cf.ngk,cf.nvt);
  switch (mode) {
    //if (mode == 0) {
    case 0:
      if (rank % 2 == 0) {
        //int temp_rank = (rank + 1 < num_procs) ? rank + 1 : 1;
        int temp_rank = 1;
        MPI_Send( a.data(), 1, rightSendSubArray, temp_rank
                , MPI_TAG2, MPI_COMM_WORLD );
        MPI_Recv( a.data(), 1, rightRecvSubArray, temp_rank
                , MPI_TAG2, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
      }
      else {
        //int temp_rank = (rank < num_procs) ? rank + 1 : 0;
        int temp_rank = 0;
        MPI_Recv( a.data(), 1, leftRecvSubArray, temp_rank
                , MPI_TAG2, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Send( a.data(), 1, leftSendSubArray, temp_rank
                , MPI_TAG2, MPI_COMM_WORLD );
      }
      break;
    //}
    case 1:
    //else if (mode == 1) {
    leftSend  = Kokkos::View<double****,FS_LAYOUT>("leftSend",
                                                   cf.ng,cf.ngj,cf.ngk,cf.nvt);
    leftRecv  = Kokkos::View<double****,FS_LAYOUT>("leftRecv",
                                                   cf.ng,cf.ngj,cf.ngk,cf.nvt);
    rightSend = Kokkos::View<double****,FS_LAYOUT>("rightSend",
                                                   cf.ng,cf.ngj,cf.ngk,cf.nvt);
    rightRecv = Kokkos::View<double****,FS_LAYOUT>("rightRecv",
                                                   cf.ng,cf.ngj,cf.ngk,cf.nvt);

      Kokkos::parallel_for( xPol, KOKKOS_LAMBDA(const int i, const int j, 
    					    const int k, const int v) {
            leftSend( i, j, k, v ) = a( cf.ng + i, j, k, v );
            rightSend( i, j, k, v ) = a( i + cf.nci, j, k, v );
          });
      Kokkos::fence();
    
      if (rank % 2 == 0) {
        //int temp_rank = (rank < num_procs) ? rank + 1 : 1;
        int temp_rank = 1;
        MPI_Send( rightSend.data(), cf.ng*cf.ngj*cf.ngk*(cf.nvt), rightSendSubArray
                , temp_rank, MPI_TAG2, MPI_COMM_WORLD );
        MPI_Recv( rightRecv.data(), cf.ng*cf.ngj*cf.ngk*(cf.nvt), rightRecvSubArray
                , temp_rank, MPI_TAG2, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
      }
      else {
        int temp_rank = 0;
        MPI_Recv( leftRecv.data(), cf.ng*cf.ngj*cf.ngk*(cf.nvt), leftRecvSubArray
                , temp_rank, MPI_TAG2, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Send( leftSend.data(), cf.ng*cf.ngj*cf.ngk*(cf.nvt), leftSendSubArray
                , temp_rank, MPI_TAG2, MPI_COMM_WORLD );
      }
    
      Kokkos::parallel_for(
          xPol, KOKKOS_LAMBDA(const int i, const int j, const int k, const int v) {
            a(i, j, k, v) = leftRecv(i, j, k, v);
            a(cf.nci - cf.ng + i, j, k, v) = rightRecv(i, j, k, v);
          });
      Kokkos::fence();
      break;
    //}
    case 2:
    //else if (mode == 2) {
      //leftSend  = Kokkos::View<double****,FS_LAYOUT>("leftSend",
      //                                               cf.ng,cf.ngj,cf.ngk,cf.nvt);
      //leftRecv  = Kokkos::View<double****,FS_LAYOUT>("leftRecv",
      //                                               cf.ng,cf.ngj,cf.ngk,cf.nvt);
      //rightSend = Kokkos::View<double****,FS_LAYOUT>("rightSend",
      //                                               cf.ng,cf.ngj,cf.ngk,cf.nvt);
      //rightRecv = Kokkos::View<double****,FS_LAYOUT>("rightRecv",
      //                                               cf.ng,cf.ngj,cf.ngk,cf.nvt);

      Kokkos::parallel_for( xPol, KOKKOS_LAMBDA(const int i, const int j, 
            				    const int k, const int v) {
            leftSend(i, j, k, v) = a(cf.ng + i, j, k, v);
            rightSend(i, j, k, v) = a(i + cf.nci, j, k, v);
          });
  
      Kokkos::deep_copy(  leftSend_H,  leftSend );
      Kokkos::deep_copy( rightSend_H, rightSend );
  
      //if (rank == 0)
      //  start = std::chrono::high_resolution_clock::now(); 
      if (rank % 2 == 0) {
        int temp_rank = 1;
        MPI_Send( rightSend_H.data(), cf.ng*cf.ngj*cf.ngk*(cf.nvt), rightSendSubArray
                , temp_rank, MPI_TAG2, MPI_COMM_WORLD );
        MPI_Recv( rightRecv_H.data(), cf.ng*cf.ngj*cf.ngk*(cf.nvt), rightRecvSubArray
                , temp_rank, MPI_TAG2, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
      }
      else {
        int temp_rank = 0;
        MPI_Recv( leftRecv_H.data(), cf.ng*cf.ngj*cf.ngk*(cf.nvt), leftRecvSubArray
                , temp_rank, MPI_TAG2, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Send( leftSend_H.data(), cf.ng*cf.ngj*cf.ngk*(cf.nvt), leftSendSubArray
                , temp_rank, MPI_TAG2, MPI_COMM_WORLD );
      }
      Kokkos::deep_copy(  leftRecv_H,  leftRecv );
      Kokkos::deep_copy( rightRecv_H, rightRecv );
  
      Kokkos::parallel_for(
          xPol, KOKKOS_LAMBDA(const int i, const int j, const int k, const int v) {
            a(i, j, k, v) = leftRecv(i, j, k, v);
            a(cf.nci - cf.ng + i, j, k, v) = rightRecv(i, j, k, v);
          });
      Kokkos::fence();
      break;
    //}
    default:
    //else {
      cout << "Invalid Directive";
      break;
  }
}

void ping_pong_n_dim( int max_i, int n_iterations, int dimension, int mode ) {

  struct inputConfig cf = executeConfiguration();
  //std::chrono::high_resolution_clock::time_point start 
  auto start = std::chrono::high_resolution_clock::now(); 
  //std::chrono::high_resolution_clock::time_point stop 
  auto stop  = std::chrono::high_resolution_clock::now();
  //std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  //std::chrono::steady_clock::time_point stop  = std::chrono::steady_clock::now();

  float duration  = std::chrono::duration<float, std::nano>(stop - start).count();
  float latency   = duration / 2;
  float bandwidth = 10 / duration;   

  FS4D a    = Kokkos::View<double ****, FS_LAYOUT>( "data", cf.ngi, cf.ngj, cf.ngk, cf.nvt );
  FS1D aR   = Kokkos::View<double    *, FS_LAYOUT>( "recieve", cf.ng * cf.ngj * cf.ngk * cf.nvt );
  FS1D aS   = Kokkos::View<double    *, FS_LAYOUT>( "send"   , cf.ng * cf.ngj * cf.ngk * cf.nvt );
  //aR_H = Kokkos::create_mirror_view(aR);
  //aS_H = Kokkos::create_mirror_view(aS);

  int rank, num_procs;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &num_procs );

  MPI_Comm node_comm;
  MPI_Comm_split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                       MPI_INFO_NULL, &node_comm );
  int node_size, node_rank;
  MPI_Comm_rank( node_comm, &node_rank );
  MPI_Comm_size( node_comm, &node_size );
  MPI_Comm_free( &node_comm );

  int  order;

  // Need to use something else instead of FS_LAYOUT here.
  if (std::is_same<FS_LAYOUT, Kokkos::LayoutLeft>::value) {
    order = MPI_ORDER_FORTRAN;
  } else if (std::is_same<FS_LAYOUT, Kokkos::LayoutRight>::value) {
    order = MPI_ORDER_C;
  } else {
    cerr << "Invalid array order in mpiBuffers.\n";
    exit(-1);
  }

  int bigsizes[4]  = { cf.ngi, cf.ngj, cf.ngk, cf.nvt };
  int xsubsizes[4] = { cf.ng,  cf.ngj, cf.ngk, cf.nvt };

  int leftRecvStarts[4]   = { 0, 0, 0, 0 };
  int leftSendStarts[4]   = { cf.ng, 0, 0, 0 };
  int rightRecvStarts[4]  = { cf.ngi - cf.ng, 0, 0, 0 };
  int rightSendStarts[4]  = { cf.nci, 0, 0, 0 };

  MPI_Type_create_subarray( 4, bigsizes, xsubsizes, leftRecvStarts
                          , order, MPI_DOUBLE, &leftRecvSubArray );
  MPI_Type_commit( &leftRecvSubArray );

  MPI_Type_create_subarray( 4, bigsizes, xsubsizes, leftSendStarts
                          , order, MPI_DOUBLE, &leftSendSubArray );
  MPI_Type_commit( &leftSendSubArray );

  MPI_Type_create_subarray( 4, bigsizes, xsubsizes, rightRecvStarts
                          , order, MPI_DOUBLE, &rightRecvSubArray );
  MPI_Type_commit( &rightRecvSubArray );

  MPI_Type_create_subarray( 4, bigsizes, xsubsizes, rightSendStarts
                          , order, MPI_DOUBLE, &rightSendSubArray );
  MPI_Type_commit( &rightSendSubArray );

  //int num_gpus;
  //cudaGetDeviceCount( &num_gpus );

  //bool gpu_send,
  //     gpu_copy,
  //     gpu_pack;

  if (rank == 0)
    start = std::chrono::high_resolution_clock::now(); 

  for (int i = 0; i < n_iterations; i++) {
    send_recv( rank, n_iterations, a, aR, aS, cf, mode );
    //send_recv( rank, n_iterations, a, aR, aS, aR_H, aS_H, cf );
  }
  if (rank == 0) {
    stop = std::chrono::high_resolution_clock::now();

    duration  = std::chrono::duration<float>( stop - start ).count();
    latency   = duration / ( n_iterations * 2 );
    bandwidth = ( cf.ng * cf.ngj * cf.ngk * cf.nvt * 8 * 2 * n_iterations ) / duration;
    //auto duration = std::chrono::duration_cast<microseconds>(stop - start); 
    cout << duration  << endl;
    cout << latency   << endl;
    cout << bandwidth << endl;
  }
}
