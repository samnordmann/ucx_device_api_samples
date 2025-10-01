UCX_PREFIX ?= /opt/ucx/install
MPI_CXX    ?= mpicxx
NVCC       ?= nvcc
CUDA_HOME  ?= /usr/local/cuda
DOCA_HOME  ?= /opt/mellanox/doca

TARGET     := alltoallv_ucx_device
SRCS       := main.cu
OBJS       := $(SRCS:.cu=.o)

INC_FLAGS  := -I$(UCX_PREFIX)/include -I$(DOCA_HOME)/include -I$(CUDA_HOME)/include
LIB_FLAGS  := -L$(UCX_PREFIX)/lib -L$(CUDA_HOME)/lib64 -lucp -luct -lucs -lcudart

RPATH_FLAGS := -Wl,-rpath,$(UCX_PREFIX)/lib -Wl,-rpath,$(CUDA_HOME)/lib64

CXXFLAGS   ?= -O2 -g
NVCCFLAGS  ?= -O2 -g -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr
# Use MPI wrapper as host compiler so mpi.h is found during NVCC host compilation
NVCCFLAGS  += -ccbin $(MPI_CXX)
## Build only for NVIDIA Ampere and Hopper
NVCCFLAGS  += -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90

all: $(TARGET)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INC_FLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(MPI_CXX) $(CXXFLAGS) -o $@ $^ $(LIB_FLAGS) $(RPATH_FLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)




