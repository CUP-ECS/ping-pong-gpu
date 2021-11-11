#include <pthread.h>
#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>

{{fn name MPI_Init}}
  nvtxNameCategoryA(999, "MPI");
  {{callfn}}
  int rank;
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char name[256];
  sprintf( name, "MPI Rank %d", rank );

  nvtxNameOsThread(pthread_self(), name);
  nvtxNameCudaDeviceA(rank, name);
{{endfn}}

{{fn name MPI_Comm_rank MPI_Comm_size MPI_Isend MPI_Irecv MPI_Type_commit MPI_Send MPI_Recv MPI_Cart_create MPI_Cart_coords MPI_Cart_shift MPI_Waitall MPI_Type_create_subarray MPI_Comm_split_type MPI_Comm_free MPI_Finalize}}
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "{{name}}";
  eventAttrib.category = 999;

  nvtxRangePushEx(&eventAttrib);
  {{callfn}}
  nvtxRangePop();
{{endfn}}
