#include "ping_pong.hpp"
#include "input.hpp"
//#include "Kokkos_Core.hpp"
//#include "kokkosTypes.hpp"
//
//Maybe pass in dimensional array

void ping_pong_n_dim( struct inputConfig &cf, int max_i,
                      int n_iterations      , int dimension ) {
  int rank, num_procs;
  int    i, j, k, v;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &num_procs );

  MPI_Comm node_comm;
  MPI_Comm_split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                       MPI_INFO_NULL, &node_comm
                     );
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
  a = Kokkos::View<double ****, FS_LAYOUT>( "send", i, j, 2, 2 );

  int bigsizes[4]  = { cf.ngi, cf.ngj, cf.ngk, cf.nvt };

  int xsubsizes[4] = { cf.ng,  cf.ngj, cf.ngk, cf.nvt };
  int ysubsizes[4] = { cf.ngi,  cf.ng, cf.ngk, cf.nvt };
  int zsubsizes[4] = { cf.ngi,  cf.ngj, cf.ng, cf.nvt };

  //aH = Kokkos::create_mirror_view(a);

  FS4D &aR = a;

  if (rank % 2 == 0) {
    MPI_Irecv( aH.data()
             , 4
             , MPI_DOUBLE
             ,
             );
  }
  else {
    MPI_Isend( aH.data()
             , 4
             , MPI_DOUBLE
             , rank + 1
             , MPI_ANY_TAG
             , MPI_COMM_WORLD
             , MPI_REQUEST_NULL
             );
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
