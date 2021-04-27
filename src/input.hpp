#include <stdio.h>
#include "mpi.h"
#include "kokkos/core/src/Kokkos_Core.hpp"
#include "KokkosTypes.hpp"

typedef typename Kokkos::View<double ****, FS_LAYOUT> FS4D;
typedef typename Kokkos::View<double ****, FS_LAYOUT>::HostMirror FS4DH;

struct inputConfig {
  int nci, ncj, nck;
  int ng, ngi, ngj, ngk, nvt;
};

class mpiHaloExchange {
  public:
    virtual void sendHalo(MPI_Request reqs[]) = 0;
    virtual void receiveHalo(MPI_Request reqs[]) = 0;
    virtual void unpackHalo() { };

    FS4D &deviceV;
    struct inputConfig &cf;
    void haloExchange();

    mpiHaloExchange(struct inputConfig &c, FS4D &v) : cf(c), deviceV(v) { };
};

class directHaloExchange : public mpiHaloExchange
{
  public:

#ifndef DIRECT
#define DIRECT

    MPI_Datatype leftRecvSubArray, rightRecvSubArray;
    MPI_Datatype leftSendSubArray, rightSendSubArray;

    virtual void sendHalo(MPI_Request reqs[]);
    virtual void receiveHalo(MPI_Request reqs[]);
    directHaloExchange(inputConfig &c, FS4D &v);
};

class packedHaloExchange : public mpiHaloExchange
{

  public:

#ifndef CUDA_AWARE
#define CUDA_AWARE

    FS4D leftSend, leftRecv;
    FS4D rightSend, rightRecv;

    // Tags and tag-dispatched operators to get around nvcc lambda capture problems
    struct xPack {};
    struct yPack {};
    struct zPack {};
    struct xUnpack {};
    struct yUnpack {};
    struct zUnpack {};

    KOKKOS_INLINE_FUNCTION
    void operator()( const xPack&, const int i, const int j, const int k, const int v ) const;
    KOKKOS_INLINE_FUNCTION
    void operator()( const yPack&, const int i, const int j, const int k, const int v ) const;
    KOKKOS_INLINE_FUNCTION
    void operator()( const zPack&, const int i, const int j, const int k, const int v ) const;

    KOKKOS_INLINE_FUNCTION
    void operator()( const xUnpack&, const int i, const int j, const int k, const int v ) const;
    KOKKOS_INLINE_FUNCTION
    void operator()( const yUnpack&, const int i, const int j, const int k, const int v ) const;
    KOKKOS_INLINE_FUNCTION
    void operator()( const zUnpack&, const int i, const int j, const int k, const int v ) const;

    virtual void sendHalo(MPI_Request reqs[]);
    virtual void receiveHalo(MPI_Request reqs[]);
    virtual void unpackHalo();

    packedHaloExchange(inputConfig &c, FS4D &v);
};

class copyHaloExchange : public packedHaloExchange
{
  public:

#ifndef COPY
#define COPY

    FS4DH leftSend_H, leftRecv_H;
    FS4DH rightSend_H, rightRecv_H;

    virtual void sendHalo(MPI_Request reqs[]);
    virtual void receiveHalo(MPI_Request reqs[]);
    virtual void unpackHalo();

    //copyHaloExchange(inputConfig &c, FS4D &v);
};
