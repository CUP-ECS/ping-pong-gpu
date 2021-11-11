
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef _EXTERN_C_
#ifdef __cplusplus
#define _EXTERN_C_ extern "C"
#else /* __cplusplus */
#define _EXTERN_C_
#endif /* __cplusplus */
#endif /* _EXTERN_C_ */

#ifdef MPICH_HAS_C2F
_EXTERN_C_ void *MPIR_ToPointer(int);
#endif // MPICH_HAS_C2F

#ifdef PIC
/* For shared libraries, declare these weak and figure out which one was linked
   based on which init wrapper was called.  See mpi_init wrappers.  */
#pragma weak pmpi_init
#pragma weak PMPI_INIT
#pragma weak pmpi_init_
#pragma weak pmpi_init__
#endif /* PIC */

_EXTERN_C_ void pmpi_init(MPI_Fint *ierr);
_EXTERN_C_ void PMPI_INIT(MPI_Fint *ierr);
_EXTERN_C_ void pmpi_init_(MPI_Fint *ierr);
_EXTERN_C_ void pmpi_init__(MPI_Fint *ierr);

#include <pthread.h>
#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>

/* ================== C Wrappers for MPI_Init ================== */
_EXTERN_C_ int PMPI_Init(int *argc, char ***argv);
_EXTERN_C_ int MPI_Init(int *argc, char ***argv) { 
    int _wrap_py_return_val = 0;

  nvtxNameCategoryA(999, "MPI");
  _wrap_py_return_val = PMPI_Init(argc, argv);
  int rank;
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char name[256];
  sprintf( name, "MPI Rank %d", rank );

  nvtxNameOsThread(pthread_self(), name);
  nvtxNameCudaDeviceA(rank, name);
    return _wrap_py_return_val;
}



/* ================== C Wrappers for MPI_Comm_rank ================== */
_EXTERN_C_ int PMPI_Comm_rank(MPI_Comm comm, int *rank);
_EXTERN_C_ int MPI_Comm_rank(MPI_Comm comm, int *rank) { 
    int _wrap_py_return_val = 0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Comm_rank";
  eventAttrib.category = 999;

  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Comm_rank(comm, rank);
  nvtxRangePop();
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Comm_size ================== */
_EXTERN_C_ int PMPI_Comm_size(MPI_Comm comm, int *size);
_EXTERN_C_ int MPI_Comm_size(MPI_Comm comm, int *size) { 
    int _wrap_py_return_val = 0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Comm_size";
  eventAttrib.category = 999;

  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Comm_size(comm, size);
  nvtxRangePop();
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Isend ================== */
_EXTERN_C_ int PMPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
_EXTERN_C_ int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) { 
    int _wrap_py_return_val = 0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Isend";
  eventAttrib.category = 999;

  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
  nvtxRangePop();
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Irecv ================== */
_EXTERN_C_ int PMPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
_EXTERN_C_ int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request) { 
    int _wrap_py_return_val = 0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Irecv";
  eventAttrib.category = 999;

  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
  nvtxRangePop();
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Type_commit ================== */
_EXTERN_C_ int PMPI_Type_commit(MPI_Datatype *datatype);
_EXTERN_C_ int MPI_Type_commit(MPI_Datatype *datatype) { 
    int _wrap_py_return_val = 0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Type_commit";
  eventAttrib.category = 999;

  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Type_commit(datatype);
  nvtxRangePop();
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Send ================== */
_EXTERN_C_ int PMPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
_EXTERN_C_ int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) { 
    int _wrap_py_return_val = 0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Send";
  eventAttrib.category = 999;

  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Send(buf, count, datatype, dest, tag, comm);
  nvtxRangePop();
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Recv ================== */
_EXTERN_C_ int PMPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
_EXTERN_C_ int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) { 
    int _wrap_py_return_val = 0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Recv";
  eventAttrib.category = 999;

  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Recv(buf, count, datatype, source, tag, comm, status);
  nvtxRangePop();
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Cart_create ================== */
_EXTERN_C_ int PMPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm *comm_cart);
_EXTERN_C_ int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm *comm_cart) { 
    int _wrap_py_return_val = 0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Cart_create";
  eventAttrib.category = 999;

  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Cart_create(comm_old, ndims, dims, periods, reorder, comm_cart);
  nvtxRangePop();
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Cart_coords ================== */
_EXTERN_C_ int PMPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]);
_EXTERN_C_ int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]) { 
    int _wrap_py_return_val = 0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Cart_coords";
  eventAttrib.category = 999;

  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Cart_coords(comm, rank, maxdims, coords);
  nvtxRangePop();
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Cart_shift ================== */
_EXTERN_C_ int PMPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rank_source, int *rank_dest);
_EXTERN_C_ int MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rank_source, int *rank_dest) { 
    int _wrap_py_return_val = 0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Cart_shift";
  eventAttrib.category = 999;

  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Cart_shift(comm, direction, disp, rank_source, rank_dest);
  nvtxRangePop();
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Waitall ================== */
_EXTERN_C_ int PMPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]);
_EXTERN_C_ int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]) { 
    int _wrap_py_return_val = 0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Waitall";
  eventAttrib.category = 999;

  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Waitall(count, array_of_requests, array_of_statuses);
  nvtxRangePop();
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Type_create_subarray ================== */
_EXTERN_C_ int PMPI_Type_create_subarray(int ndims, const int array_of_sizes[], const int array_of_subsizes[], const int array_of_starts[], int order, MPI_Datatype oldtype, MPI_Datatype *newtype);
_EXTERN_C_ int MPI_Type_create_subarray(int ndims, const int array_of_sizes[], const int array_of_subsizes[], const int array_of_starts[], int order, MPI_Datatype oldtype, MPI_Datatype *newtype) { 
    int _wrap_py_return_val = 0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Type_create_subarray";
  eventAttrib.category = 999;

  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes, array_of_starts, order, oldtype, newtype);
  nvtxRangePop();
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Comm_split_type ================== */
_EXTERN_C_ int PMPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm);
_EXTERN_C_ int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm) { 
    int _wrap_py_return_val = 0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Comm_split_type";
  eventAttrib.category = 999;

  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Comm_split_type(comm, split_type, key, info, newcomm);
  nvtxRangePop();
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Comm_free ================== */
_EXTERN_C_ int PMPI_Comm_free(MPI_Comm *comm);
_EXTERN_C_ int MPI_Comm_free(MPI_Comm *comm) { 
    int _wrap_py_return_val = 0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Comm_free";
  eventAttrib.category = 999;

  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Comm_free(comm);
  nvtxRangePop();
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Finalize ================== */
_EXTERN_C_ int PMPI_Finalize();
_EXTERN_C_ int MPI_Finalize() { 
    int _wrap_py_return_val = 0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Finalize";
  eventAttrib.category = 999;

  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Finalize();
  nvtxRangePop();
    return _wrap_py_return_val;
}


