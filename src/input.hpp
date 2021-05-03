#include <stdio.h>
#include "mpi.h"
#include "Kokkos_Core.hpp"
#include "KokkosTypes.hpp"

struct inputConfig executeConfiguration();

//FS4D &deviceV;
//struct inputConfig &cf;
void haloExchange();

MPI_Datatype leftRecvSubArray, rightRecvSubArray;
MPI_Datatype leftSendSubArray, rightSendSubArray;

FS4D leftSend, leftRecv;
FS4D rightSend, rightRecv;

struct inputConfig {
  int nci, ncj, nck;
  int ng, ngi, ngj, ngk, nvt;
};

#ifndef DIRECT
#define DIRECT
#endif

#ifndef CUDA_AWARE
#define CUDA_AWARE
    
    // Tags and tag-dispatched operators to get around nvcc lambda capture problems
    struct xPack {};
    struct yPack {};
    struct zPack {};
    struct xUnpack {};
    struct yUnpack {};
    struct zUnpack {};
    
//    KOKKOS_INLINE_FUNCTION
//    void operator()( const xPack&, const int i, const int j
//                   , const int k, const int v ) const;
//    KOKKOS_INLINE_FUNCTION
//    void operator()( const yPack&, const int i, const int j
//                   , const int k, const int v ) const;
//    KOKKOS_INLINE_FUNCTION
//    void operator()( const zPack&, const int i, const int j
//                   , const int k, const int v ) const;
//    KOKKOS_INLINE_FUNCTION
//    void operator()( const xUnpack&, const int i, const int j
//                   , const int k, const int v ) const;
//    KOKKOS_INLINE_FUNCTION
//    void operator()( const yUnpack&, const int i, const int j
//                   , const int k, const int v ) const;
//    KOKKOS_INLINE_FUNCTION
//    void operator()( const zUnpack&, const int i, const int j
//                   , const int k, const int v ) const;

#endif

#ifndef COPY
#define COPY

    FS4DH leftSend_H, leftRecv_H;
    FS4DH rightSend_H, rightRecv_H;

#endif
