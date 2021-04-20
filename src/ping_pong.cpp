#include "ping_pong.hpp"
#include "input.hpp"
//#include "Kokkos_Core.hpp"
//#include "kokkosTypes.hpp"
//
//Maybe pass in dimensional array

void ping_pong_n_dim( struct inputConfig &cf, int max_i,
                      int n_iterations      , int dimension ) {
  int rank, num_procs;
  int i, j, k, v;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &num_procs );

  MPI_Comm node_comm;
  MPI_Comm_split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                       MPI_INFO_NULL, &node_comm );
  int node_size, node_rank;
  MPI_Comm_rank( node_comm, &node_rank );
  MPI_Comm_size( node_comm, &node_size );
  MPI_Comm_free( &node_comm );

  int num_gpus;
  cudaGetDeviceCount( &num_gpus );

  int  max_bytes = pow( 2, max_i - 1 ) * sizeof( float );
  int  dim_collapse;
  int  collapse_size;
  bool gpu_send,
       gpu_copy,
       gpu_pack;

  double time, max_time;
  bool active;

#ifdef DIRECT
  int order;
  if (std::is_same<FS_LAYOUT, Kokkos::LayoutLeft>::value) {
    order = MPI_ORDER_FORTRAN;
  } else if (std::is_same<FS_LAYOUT, Kokkos::LayoutRight>::value) {
    order = MPI_ORDER_C;
  } else {
    cerr << "Invalid array order in mpiBuffers.\n";
    exit(-1);
  }
  a = Kokkos::View<double ****, FS_LAYOUT>( "send", i, j, 2, 2 );

  int bigsizes[4]  = { cf.ngi, cf.ngj, cf.ngk, cf.nvt };

  int xsubsizes[4] = { cf.ng,  cf.ngj, cf.ngk, cf.nvt };
  int ysubsizes[4] = { cf.ngi,  cf.ng, cf.ngk, cf.nvt };
  int zsubsizes[4] = { cf.ngi,  cf.ngj, cf.ng, cf.nvt };

  int leftRecvStarts[4]   = { 0, 0, 0, 0 };
  int leftSendStarts[4]   = { cf.ng, 0, 0, 0 };
  int rightRecvStarts[4]  = { cf.ngi - cf.ng, 0, 0, 0 };
  int rightSendStarts[4]  = { cf.nci, 0, 0, 0 };
  int bottomRecvStarts[4] = { 0, 0, 0, 0 };
  int bottomSendStarts[4] = { 0, cf.ng, 0, 0 };
  int topRecvStarts[4]    = { 0, cf.ngj - cf.ng, 0, 0 };
  int topSendStarts[4]    = { 0, cf.ncj, 0, 0 };
  //aH = Kokkos::create_mirror_view(a);

  MPI_Type_create_subarray( 4, bigsizes, xsubsizes, leftRecvStarts
                          , order, MPI_DOUBLE, &leftRecvSubArray );
  MPI_Type_commit(&leftRecvSubArray);

  MPI_Type_create_subarray(4, bigsizes, xsubsizes, leftSendStarts
                          , order, MPI_DOUBLE, &leftSendSubArray);
  MPI_Type_commit(&leftSendSubArray);

  MPI_Type_create_subarray(4, bigsizes, xsubsizes, rightRecvStarts
                          , order, MPI_DOUBLE, &rightRecvSubArray);
  MPI_Type_commit(&rightRecvSubArray);

  MPI_Type_create_subarray(4, bigsizes, xsubsizes, rightSendStarts
                          , order, MPI_DOUBLE, &rightSendSubArray);
  MPI_Type_commit(&rightSendSubArray);

  MPI_Type_create_subarray(4, bigsizes, ysubsizes, bottomRecvStarts
                          , order, MPI_DOUBLE, &bottomRecvSubArray);
  MPI_Type_commit(&bottomRecvSubArray);

  MPI_Type_create_subarray(4, bigsizes, ysubsizes, bottomSendStarts
                          , order, MPI_DOUBLE, &bottomSendSubArray);
  MPI_Type_commit(&bottomSendSubArray);

  MPI_Type_create_subarray(4, bigsizes, ysubsizes, topRecvStarts
                          , order, MPI_DOUBLE, &topRecvSubArray);
  MPI_Type_commit(&topRecvSubArray);

  MPI_Type_create_subarray(4, bigsizes, ysubsizes, topSendStarts
                          , order, MPI_DOUBLE, &topSendSubArray);
  MPI_Type_commit(&topSendSubArray);

  FS4D &aR = a;

  if (rank % 2 == 0) {
    int temp_rank = (rank < num_procs) ? rank + 1 : 1;
    MPI_Irecv( a.data(), 1, leftRecvSubArray  , temp_rank
             , MPI_ANY_TAG, MPI_COMM_WORLD, MPI_REQUEST_NULL );
    MPI_Irecv( a.data(), 1, rightRecvSubArray , temp_rank
             , MPI_ANY_TAG, MPI_COMM_WORLD, MPI_REQUEST_NULL );
    MPI_Irecv( a.data(), 1, bottomRecvSubArray, temp_rank
             , MPI_ANY_TAG, MPI_COMM_WORLD, MPI_REQUEST_NULL );
    MPI_Irecv( a.data(), 1, topRecvSubArray   , temp_rank
             , MPI_ANY_TAG, MPI_COMM_WORLD, MPI_REQUEST_NULL );
  }
  else {
    int temp_rank = (rank < num_procs) ? rank + 1 : 0;
    MPI_Isend( a.data(), 1, leftSendSubArray  , temp_rank
             , MPI_ANY_TAG, MPI_COMM_WORLD, MPI_REQUEST_NULL );
    MPI_Isend( a.data(), 1, rightSendSubArray , temp_rank
             , MPI_ANY_TAG, MPI_COMM_WORLD, MPI_REQUEST_NULL );
    MPI_Isend( a.data(), 1, bottomSendSubArray, temp_rank
             , MPI_ANY_TAG, MPI_COMM_WORLD, MPI_REQUEST_NULL );
    MPI_Isend( a.data(), 1, topSendSubArray   , temp_rank
             , MPI_ANY_TAG, MPI_COMM_WORLD, MPI_REQUEST_NULL );
           //  MPI_Isend( aH.data()
  }
//           , 4
//           , MPI_DOUBLE
//           ,
//           );
//  MPI_Irecv( aH.data()
//           , 4
//           , MPI_DOUBLE
//           ,
//           )

#endif
#ifdef CUDA_AWARE
  a = Kokkos::View<double ****, FS_LAYOUT>("send", i, j, 2, 2);

  aH = Kokkos::create_mirror_view(a);

  FS4D &aR = a;

  Kokkos::deep_copy(aH, a);

  MPI_Isend( aH.data()
           , 4
           , MPI_DOUBLE
           ,
           )
  MPI_Irecv( aH.data()
           , 4
           , MPI_DOUBLE
           ,
           )

#endif
#ifdef COPY
  a = Kokkos::View<double ****, FS_LAYOUT>("send", i, j, 2, 2);

  aH = Kokkos::create_mirror_view(a);

  FS4D &aR = a;

  MPI_Isend( aH.data()
           , 4
           , MPI_DOUBLE
           ,
           )
  MPI_Irecv( aH.data()
           , 4
           , MPI_DOUBLE
           ,
           )

#endif

}
