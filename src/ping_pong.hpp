#include "Kokkos_Core.hpp"
#include "KokkosTypes.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define FS_LAYOUT Kokkos::DefaultExecutionSpace::array_layout
#define MPI_TAG2 4

typedef typename Kokkos::View<double ****, FS_LAYOUT> FS4D;
typedef typename Kokkos::View<double ****, FS_LAYOUT>::HostMirror FS4DH;

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

struct inputConfig {
  int nci, ncj, nck;
  int ng, ngi, ngj, ngk, nvt;
};

class mpiHaloExchange {
  public:
    //virtual void sendHalo(MPI_Request reqs[]) = 0;
    //virtual void receiveHalo(MPI_Request reqs[]) = 0;
    //virtual void unpackHalo() { };

    FS4D &deviceV;
    struct inputConfig &cf;
    //void haloExchange();

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
