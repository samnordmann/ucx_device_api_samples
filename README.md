## UCX device-side API GPU-to-GPU example (CUDA + MPI)

This example performs an alltoallv using the UCX device-side API from a CUDA kernel. Each rank sends a variable-length segment to every other rank. The kernel issues device-side PUTs to all peers.

### Prerequisites
- UCX built with CUDA and device-side API support (headers and libs available)
- OpenMPI
- Cuda

### Build
```bash
make
```

### Run
```bash
mpirun -np 4 ./alltoallv_ucx_device
```



