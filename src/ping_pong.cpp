#include "ping_pong.hpp"
//#include "input.hpp"
#include <iostream>
#include <fstream>
#include "Kokkos_Core.hpp"
#include "KokkosTypes.hpp"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

using namespace std;

//#define DIRECT
//#define CUDA_AWARE
//#define COPY

//FS4D leftSend, leftRecv;
//FS4D rightSend, rightRecv;


//mpiBuffers::mpiBuffers(struct inputConfig cf) {
//
//}

void direct( int rank, int n_iterations, FS4D a
           , inputConfig cf, int mode, int order
           , MPI_Datatype leftRecvSubArray, MPI_Datatype rightRecvSubArray
           , MPI_Datatype leftSendSubArray, MPI_Datatype rightSendSubArray
           ) {
  if (rank % 2 == 0) {
    int temp_rank = rank + 1;
    MPI_Send( a.data(), 1, rightSendSubArray, temp_rank
            , MPI_TAG2, MPI_COMM_WORLD );
    MPI_Recv( a.data(), 1, rightRecvSubArray, temp_rank
            , MPI_TAG2, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
  }
  else {
    int temp_rank = rank - 1;
    MPI_Recv( a.data(), 1, leftRecvSubArray, temp_rank
            , MPI_TAG2, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    MPI_Send( a.data(), 1, leftSendSubArray, temp_rank
            , MPI_TAG2, MPI_COMM_WORLD );
  }
}

void cuda_pack( int rank, int n_iterations, FS4D a, inputConfig cf
              , int mode, int order, FS4D send, FS4D recv, FS4DH send_H
              , FS4DH recv_H, int direction, int copy
               ) {
  auto xPol = Kokkos::MDRangePolicy<Kokkos::Rank<4>>( {0, 0, 0, 0},
                                            {cf.ng, cf.ngj, cf.ngk, cf.nvt} );
  auto yPol = Kokkos::MDRangePolicy<Kokkos::Rank<4>>( {0, 0, 0, 0},
                                            {cf.ngi, cf.ng, cf.ngk, cf.nvt} );
  auto zPol = Kokkos::MDRangePolicy<Kokkos::Rank<4>>( {0, 0, 0, 0},
                                            {cf.ngi, cf.ngj, cf.ng, cf.nvt} );
  void *sendbuf, *recvbuf;

  if ( rank % 2 == 0 ) {
     switch( direction ) {
	case 0:
	   Kokkos::parallel_for(
	      xPol, KOKKOS_LAMBDA( const int i, const int j, const int k, const int v ) {
		 send( i, j, k, v ) = a( i + cf.nci, j, k, v );
	      });
	   break;
	case 1:
	   Kokkos::parallel_for(
	      yPol, KOKKOS_LAMBDA( const int i, const int j, const int k, const int v ) {
		 send( i, j, k, v ) = a( i, j + cf.ncj, k, v );
	      });
	   break;
	case 2:
	   Kokkos::parallel_for(
	      zPol, KOKKOS_LAMBDA( const int i, const int j, const int k, const int v ) {
		 send( i, j, k, v ) = a( i, j, k + cf.nck, v );
	      });
	   break;
     }
     if (copy) {
        Kokkos::deep_copy(send_H, send);
        sendbuf = send_H.data();
        recvbuf = recv_H.data();
     } else {
        sendbuf = send.data();
        recvbuf = recv.data();
     }

     Kokkos::fence();
  
     int temp_rank = rank + 1;
     MPI_Send( sendbuf, cf.ng*cf.ngj*cf.ngk*(cf.nvt), MPI_DOUBLE
             , temp_rank, MPI_TAG2, MPI_COMM_WORLD );
     MPI_Recv( recvbuf, cf.ng*cf.ngj*cf.ngk*(cf.nvt), MPI_DOUBLE
             , temp_rank, MPI_TAG2, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

     if (copy) {
        Kokkos::deep_copy(recv, recv_H);
     }

     switch( direction ) {
	case 0:
	   Kokkos::parallel_for(
	      xPol, KOKKOS_LAMBDA( const int i, const int j, const int k, const int v ) {
		    a(cf.ngi - cf.ng + i, j, k, v) = recv(i, j, k, v);
	      });
	   break;
	case 1:
	   Kokkos::parallel_for(
	      yPol, KOKKOS_LAMBDA( const int i, const int j, const int k, const int v ) {
		 a(i, cf.ngj - cf.ng + j, k, v) = recv(i, j, k, v);
	      });
	   break;
	case 2:
	   Kokkos::parallel_for(
	      zPol, KOKKOS_LAMBDA( const int i, const int j, const int k, const int v ) {
		 a(i, j, cf.ngk - cf.nk + k, v) = recv(i, j, k, v);
	      });
	   break;
     }
     // Make sure our data is in the receive array before returning and potentially 
     // starting the next iteration.
     Kokkos::fence();
  }
  else {
     int temp_rank = rank - 1;

     if (copy) {
        recvbuf = recv_H.data();
        sendbuf = send_H.data();
     } 
     else {
        recvbuf = recv.data();
        sendbuf = send.data();
     }

     MPI_Recv( recvbuf, cf.ng*cf.ngj*cf.ngk*(cf.nvt), MPI_DOUBLE
               , temp_rank, MPI_TAG2, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
     /* We split this into two parallel for loops to be fair to MPI and because
      * we're basically trying to measure one-way multi-send latency/bandwidth
      * and want each direction to do the same work */

     if (copy) {  
        Kokkos::deep_copy(recv, recv_H);
     } 

     switch( direction ) {
	case 0:
	   Kokkos::parallel_for(
	      xPol, KOKKOS_LAMBDA( const int i, const int j, const int k, const int v ) {
		 a(i, j, k, v) = recv(i, j, k, v);
	      });
	   Kokkos::parallel_for(
	      xPol, KOKKOS_LAMBDA( const int i, const int j, const int k, const int v ) {
		 send( i, j, k, v ) = a(i, j, k, v );
	      });
	   break;
	case 1:
	   Kokkos::parallel_for(
	      yPol, KOKKOS_LAMBDA( const int i, const int j, const int k, const int v ) {
		 a(i, j, k, v) = recv(i, j, k, v);
	      });
	   Kokkos::parallel_for(
	      yPol, KOKKOS_LAMBDA( const int i, const int j, const int k, const int v ) {
		 send(  i, j, k, v ) = a( i, j, k, v );
	      });
	   break;
	case 2:
	   Kokkos::parallel_for(
	      zPol, KOKKOS_LAMBDA( const int i, const int j, const int k, const int v ) {
		 a(i, j, k, v) = recv(i, j, k, v);
	      });
	   Kokkos::parallel_for(
	      zPol, KOKKOS_LAMBDA( const int i, const int j, const int k, const int v ) {
		 send(i, j, k, v ) = a( i, j, k, v );
	      });
	   break;
     }

     if (copy) {
        Kokkos::deep_copy(send_H, send);
     } else { 
        // Make sure our packing loop has finished before sending.
        Kokkos::fence(); 
     }

     // Data packed and copied if needed. Send it.
     MPI_Send( sendbuf, cf.ng*cf.ngj*cf.ngk*(cf.nvt), MPI_DOUBLE
             , temp_rank, MPI_TAG2, MPI_COMM_WORLD );
  }
}

void send_recv( int rank, int n_iterations, FS4D a, inputConfig cf
              , int mode, int order, FS4D leftSend, FS4D leftRecv, FS4D rightSend
              , FS4D rightRecv, MPI_Datatype leftRecvSubArray
              , MPI_Datatype rightRecvSubArray, MPI_Datatype leftSendSubArray
              , MPI_Datatype rightSendSubArray, FS4DH leftSend_H, FS4DH leftRecv_H
              , FS4DH rightSend_H, FS4DH rightRecv_H, int direction
              ) {

//void send_recv( int rank, int n_iterations, FS4D a, 
//                FS1D aR,  FS1D aS, inputConfig cf, int mode, int order, FS4D leftSend,
//                FS4D leftRecv, FS4D rightSend, FS4D rightRecv ) {
//void send_recv( int rank, int n_iterations, FS4D a, 
//                FS1D aR,  FS1D aS, inputConfig cf, int mode, int order ) {
  switch (mode) {
    case 0:
      direct( rank, n_iterations, a, cf, mode, order, leftRecvSubArray
            , rightRecvSubArray, leftSendSubArray, rightSendSubArray);
      break;
    case 1:
      cuda_pack( rank, n_iterations, a, cf, mode, order
                , leftSend, leftRecv, leftSend_H, leftRecv_H, direction, 0);
      break;
    case 2:
      cuda_pack( rank, n_iterations, a, cf, mode, order
                , leftSend, leftRecv, leftSend_H, leftRecv_H, direction, 1);
      break;
    default:
      cout << "Invalid Directive\n";
      break;
  }
}

void ping_pong_n_dim( int max_i, int n_iterations, int mode, int direction ) {

  ofstream file;

  MPI_Datatype leftRecvSubArray, rightRecvSubArray;
  MPI_Datatype leftSendSubArray, rightSendSubArray;

  struct inputConfig cf = executeConfiguration( max_i );
  auto start       = std::chrono::high_resolution_clock::now(); 
  auto stop        = std::chrono::high_resolution_clock::now();
  double duration  = std::chrono::duration<double, std::nano>( stop - start ).count();
  double latency   = duration / 2;
  double bandwidth = 10 / duration;

  FS4D a  = Kokkos::View<double ****, FS_LAYOUT>( "data", cf.ngi, cf.ngj, cf.ngk,  cf.nvt );

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

  int max_bytes = pow( 2, max_i - 1 ) * sizeof( float );
  int order;

  if (std::is_same<FS_LAYOUT, Kokkos::LayoutLeft>::value) {
    order = MPI_ORDER_FORTRAN;
  } else if (std::is_same<FS_LAYOUT, Kokkos::LayoutRight>::value) {
    order = MPI_ORDER_C;
  } else {
    cerr << "Invalid array order in mpiBuffers.\n";
    exit(-1);
  }

  // Set up subarray datatypes
  int bigsizes[4]  = { cf.ngi, cf.ngj, cf.ngk, cf.nvt };
  int xsubsizes[4] = { cf.ng,  cf.ngj, cf.ngk, cf.nvt };
  int ysubsizes[4] = { cf.ngi,  cf.ng, cf.ngk, cf.nvt };
  int zsubsizes[4] = { cf.ngi,  cf.ngj, cf.ng, cf.nvt };
  int leftRecvStarts[4], leftSendStart[4], rightRecvStarts[4], rightSendStarts[4];

  switch(direction) {
  case 0:
      leftRecvStarts[4]  = { 0, 0, 0, 0 };
      leftSendStarts[4]  = { cf.ng, 0, 0, 0 };
      rightRecvStarts[4] = { cf.ngi - cf.ng, 0, 0, 0 };
      rightSendStarts[4] = { cf.nci, 0, 0, 0 };
      break;
  case 1:
      leftRecvStarts[4]  = { 0, 0, 0, 0 };
      leftSendStarts[4]  = { 0, cf.ng, 0, 0 };
      rightRecvStarts[4] = { 0, cf.ngj - cf.ng, 0, 0 };
      rightSendStarts[4] = { 0, cf.ncj, 0, 0 };
      break;
  case 2:
      leftRecvStarts[4]  = { 0, 0, 0, 0 };
      leftSendStarts[4]  = { 0, 0, cf.ng, 0 };
      rightRecvStarts[4] = { 0, 0, cf.ngk - cf.ng, 0 };
      rightSendStarts[4] = { 0, 0, cf.nck, 0 };
      break;
  default:
      assert("Invalid direction argument." && 0);
      break;
  }
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

  FS4D leftSend, leftRecv, rightSend, rightRecv;
  if (direction == 0) { 
    leftSend = Kokkos::View<double****,FS_LAYOUT>("leftSend",
                                       cf.ng,cf.ngj,cf.ngk,cf.nvt);
    leftRecv  = Kokkos::View<double****,FS_LAYOUT>("leftRecv",
                                       cf.ng,cf.ngj,cf.ngk,cf.nvt);
    rightSend = Kokkos::View<double****,FS_LAYOUT>("rightSend",
                                        cf.ng,cf.ngj,cf.ngk,cf.nvt);
    rightRecv = Kokkos::View<double****,FS_LAYOUT>("rightRecv",
                                       cf.ng,cf.ngj,cf.ngk,cf.nvt);
   } else if (direction == 1) {
    leftSend = Kokkos::View<double****,FS_LAYOUT>("leftSend",
                                       cf.ngi,cf.ng,cf.ngk,cf.nvt);
    leftRecv  = Kokkos::View<double****,FS_LAYOUT>("leftRecv",
                                       cf.ngi,cf.ng,cf.ngk,cf.nvt);
    rightSend = Kokkos::View<double****,FS_LAYOUT>("rightSend",
                                        cf.ngi,cf.ng,cf.ngk,cf.nvt);
    rightRecv = Kokkos::View<double****,FS_LAYOUT>("rightRecv",
                                       cf.ngi,cf.ng,cf.ngk,cf.nvt);
  } else if (direction == 2) {
    leftSend = Kokkos::View<double****,FS_LAYOUT>("leftSend",
                                       cf.ngi,cf.ngj,cf.ng,cf.nvt);
    leftRecv  = Kokkos::View<double****,FS_LAYOUT>("leftRecv",
                                       cf.ngi,cf.ngj,cf.ng,cf.nvt);
    rightSend = Kokkos::View<double****,FS_LAYOUT>("rightSend",
                                        cf.ngi,cf.ngj,cf.ng,cf.nvt);
    rightRecv = Kokkos::View<double****,FS_LAYOUT>("rightRecv",
                                       cf.ngi,cf.ngj,cf.ng,cf.nvt);
  } else {
    assert("Invalid direction argument." && 0);
  }

  FS4DH leftSend_H  = Kokkos::create_mirror_view(leftSend); 
  FS4DH leftRecv_H  = Kokkos::create_mirror_view(leftRecv); 
  FS4DH rightSend_H = Kokkos::create_mirror_view(rightSend); 
  FS4DH rightRecv_H = Kokkos::create_mirror_view(leftRecv); 

  if ( rank == 0 )
    start = std::chrono::high_resolution_clock::now(); 

  for ( int i = 0; i < n_iterations; i++ ) {
    send_recv( rank, n_iterations, a, cf, mode, order
             , leftSend, leftRecv, rightSend, rightRecv, leftRecvSubArray, rightRecvSubArray
             , leftSendSubArray, rightSendSubArray, leftSend_H, leftRecv_H, rightSend_H
             , rightRecv_H, direction
             );
  }
  if (rank == 0) {
    stop = std::chrono::high_resolution_clock::now();

    duration  = std::chrono::duration<double>( stop - start ).count();
    latency   = duration / ( n_iterations * 2 );
    bandwidth = ( (double) cf.ng * cf.ngj * cf.ngk * cf.nvt * 8.0 * 2.0 * n_iterations ) / duration;

    switch ( mode ) {
    case 0:
        file.open("ping_pong_direct" + to_string( direction ) + ".dat", ios::out | ios::app);
        break;
    case 1:
        file.open("ping_pong_cuda" + to_string( direction ) + ".dat", ios::out | ios::app);
        break;
    case 2:
        file.open("ping_pong_copy" + to_string( direction ) + ".dat", ios::out | ios::app);
        break;       
    }

    //file << "duration,latency,bandwidth"
    file << to_string( max_i     ) + ","
          + to_string( duration  ) + "," 
          + to_string( latency   ) + "," 
          + to_string( bandwidth ) + "," 
          + to_string( direction ) << endl;

    cout << "Duration  = " + to_string( duration )  << endl;
    cout << "Latency   = " + to_string( latency )   << endl;
    cout << "Bandwidth = " + to_string( bandwidth ) << endl;

    file.close();
  }
}

