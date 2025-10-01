FROM nvcr.io/nvidia/cuda-dl-base:24.09-cuda12.6-devel-ubuntu22.04


RUN apt-get update

RUN apt-get remove -y --purge \
    libibverbs1 \
    librdmacm1 \
    librdmacm-dev

RUN wget https://www.mellanox.com/downloads/DOCA/DOCA_v3.1.0/host/doca-host_3.1.0-091000-25.07-ubuntu2204_amd64.deb && \
    dpkg -i doca-host_3.1.0-091000-25.07-ubuntu2204_amd64.deb

RUN apt-get update && apt-get install -y \
    autoconf \
    automake \
    libtool \
    python3 \
    libibverbs1 \
    librdmacm1 \
    librdmacm-dev \
    libibverbs-dev \
    rdma-core \
    libz-dev \
    libiberty-dev \
    numactl \
    libnuma-dev \
    wget \
    clangd-14 \
    bear \
    doca-sdk-gpunetio \
    libdoca-sdk-gpunetio-dev

# Install HPC-X
RUN rm -rf /opt/hpcx/ && cd /tmp && \
    wget -q http://content.mellanox.com/hpc/hpc-x/v2.23/hpcx-v2.23-gcc-inbox-ubuntu22.04-cuda12-x86_64.tbz && \
    tar xf hpcx-v2.23-gcc-inbox-ubuntu22.04-cuda12-x86_64.tbz  && \
    mv hpcx-v2.23-gcc-inbox-ubuntu22.04-cuda12-x86_64  /opt/hpcx && \
    rm -f hpcx-v2.23-gcc-inbox-ubuntu22.04-cuda12-x86_64.tbz

ENV DIR_UCX=/opt/ucx
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/openucx/ucx $DIR_UCX && \
    cd $DIR_UCX && \
    ./autogen.sh && \
    ./contrib/configure-release --prefix=$DIR_UCX/install --with-cuda=/usr/local/cuda --with-doca-gpunetio=/opt/mellanox/doca && \
    make -j && \
    make -j install
