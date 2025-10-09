#include <mpi.h>
#include <cuda_runtime.h>

#include <ucp/api/ucp.h>
#include <ucp/api/ucp_def.h>
#include <ucp/api/device/ucp_host.h>
#include <ucp/api/device/ucp_device_impl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/*
Global parameter to control whether the transfers are transported by nvLinks or by Infiniband on the scale-out network.
UCX device-side API is transparent to the transport being used, in the sense that the API semantics supports those two transports.
However, to get the best performance, the developper has to adapt the kernels to the underlying transport.
Indeed, the two transports obey to much different paradigms.

On the one hand, nvLinks transports triggered by device-side API are 
1) SM-driven, which means we need many threads to participate to the transfer to get good performance
2) blocking, meaning that we do not need to check for completion.

On the other hand, Infiniband is
1) driven by the NIC, whichmeans we need a single thread to participate to the transfer (the thread will only ring a doorbell to the NIC) ; therefore we want each thread to post a different request dso we can emit enough request to saturate the network's BW
2) non-blocking, meaning that we need to check for completion.

The API `ucp_device_put_single` take a template parameter `level` that controls the level of concurrency, which can be either of "thread" or "warp" or "block".
For example, a level "block" means that all threads in the block must participate to the same call ; "thread" level means only one thread must participate to the same call.
Typically, an nvLink transport requires a "block" level to get good performance, whereas an Infiniband transport requires a "thread".

The API `ucp_device_progress_req` is used to check for completion of a request. It is typically a trivial call in the case of an nvLink transport.
*/
constexpr bool nvLink_transport = true;

// Simple CUDA check
#define CUDA_CHECK(cmd) do { \
    cudaError_t _e = (cmd); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define MPI_CHECK(cmd) do { \
    int _e = (cmd); \
    if (_e != MPI_SUCCESS) { \
        fprintf(stderr, "MPI error %s:%d code=%d\n", __FILE__, __LINE__, _e); \
        MPI_Abort(MPI_COMM_WORLD, _e); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define UCP_CHECK(sts, msg) do { \
    if ((sts) != UCS_OK) { \
        fprintf(stderr, "UCX error %s:%d: %s failed: %d\n", __FILE__, __LINE__, (msg), (sts)); \
        MPI_Abort(MPI_COMM_WORLD, (sts)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Kernel params
typedef struct {
    unsigned                              num_threads;
    unsigned                              num_blocks;
    ucs_device_level_t                    level;
    bool                                  with_request;
    const ucp_device_mem_list_handle_h   *mem_lists;  // device pointer to array of handles
    unsigned                              num_lists;  // equals world_size
} kernel_params_t;

// A single PUT operation descriptor for device-side API
typedef struct {
    unsigned     list_handle_index; // index into params.mem_lists (per-destination)
    unsigned     element_index;     // element inside that mem list (always 0 here)
    size_t       local_offset;      // offset from base local_addr in mem list
    size_t       remote_offset;     // offset from base remote_addr in mem list
    size_t       length;            // bytes to transfer
} put_op_t;

__global__ void do_alltoallv_kernel_nvLink_transport(kernel_params_t params,
                                     const put_op_t *ops,
                                     unsigned num_ops,
                                     ucs_status_t *status_out)
{
    unsigned bid = blockIdx.x;
    if (bid >= num_ops) return;

    // All threads in the block participate in the same PUT to maximize SM-driven bandwidth
    const put_op_t &op = ops[bid];
    ucs_status_t st = ucp_device_put_single<UCS_DEVICE_LEVEL_BLOCK>(params.mem_lists[op.list_handle_index],
                                                   op.element_index,
                                                   op.local_offset,
                                                   op.remote_offset,
                                                   op.length,
                                                   /*channel_id=*/0,
                                                   UCP_DEVICE_FLAG_NODELAY,
                                                   /*req=*/nullptr);
    // Update status out
    (void)atomicCAS((int*)status_out, (int)UCS_OK, (int)st);
}

// Runs the device kernel benchmark and prints performance on rank 0
__global__ void do_alltoallv_kernel_ib_transport(kernel_params_t params,
                                     const put_op_t *ops,
                                     unsigned num_ops,
                                     ucs_status_t *status_out)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_ops) return;

    ucp_device_request_t req_obj;
    ucp_device_request_t *req = &req_obj;

    const put_op_t &op = ops[tid];
    ucs_status_t st = ucp_device_put_single<UCS_DEVICE_LEVEL_THREAD>(params.mem_lists[op.list_handle_index],
                                                   op.element_index,
                                                   op.local_offset,
                                                   op.remote_offset,
                                                   op.length,
                                                   /*channel_id=*/0,
                                                   UCP_DEVICE_FLAG_NODELAY,
                                                   req);
    if (st == UCS_OK) {
        for (;;) {
            st = ucp_device_progress_req<UCS_DEVICE_LEVEL_THREAD>(req);
            if (st != UCS_INPROGRESS) {
                break;
            }
        }
    }
    // Update status out
    (void)atomicCAS((int*)status_out, (int)UCS_OK, (int)st);
}

static void init_ucp(ucp_context_h &ucp_ctx, ucp_worker_h &worker)
{
    ucp_params_t ucp_params;
    memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features   = UCP_FEATURE_RMA | UCP_FEATURE_DEVICE;
    UCP_CHECK(ucp_init(&ucp_params, nullptr, &ucp_ctx), "ucp_init");

    ucp_worker_params_t worker_params;
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    UCP_CHECK(ucp_worker_create(ucp_ctx, &worker_params, &worker), "ucp_worker_create");
}

static void create_all_endpoints(ucp_worker_h worker, int rank, int size, std::vector<ucp_ep_h> &eps)
{
    // Gather all worker addresses
    ucp_address_t *my_addr; size_t my_addr_len;
    UCP_CHECK(ucp_worker_get_address(worker, &my_addr, &my_addr_len), "ucp_worker_get_address");

    std::vector<size_t> addr_lens(size);
    MPI_CHECK(MPI_Allgather(&my_addr_len, 1, MPI_UNSIGNED_LONG, addr_lens.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD));

    // Convert size_t to int for MPI_Allgatherv
    std::vector<int> addr_lens_i(addr_lens.begin(), addr_lens.end());

    size_t total = 0; for (auto l : addr_lens) total += l;
    std::vector<uint8_t> all_addr(total);

    std::vector<int> displs(size, 0);
    for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + addr_lens_i[i-1];

    MPI_CHECK(MPI_Allgatherv(my_addr, my_addr_len, MPI_BYTE,
                             all_addr.data(), addr_lens_i.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD));

    eps.assign(size, nullptr);
    for (int peer = 0; peer < size; ++peer) {
        if (peer == rank) continue;
        const uint8_t *peer_addr_ptr = all_addr.data() + displs[peer];
        ucp_ep_params_t ep_params; memset(&ep_params, 0, sizeof(ep_params));
        ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        ep_params.address    = (const ucp_address_t*)peer_addr_ptr;
        UCP_CHECK(ucp_ep_create(worker, &ep_params, &eps[peer]), "ucp_ep_create");
    }

    ucp_worker_release_address(worker, my_addr);
}


static void run_device_kernel_benchmark(int rank,
                                        int world_size,
                                        const kernel_params_t &kparams,
                                        const put_op_t *d_ops,
                                        unsigned num_ops,
                                        ucs_status_t *d_status,
                                        ucs_status_t &h_status,
                                        const std::vector<int> &sendcounts)
{
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const int warmup_iters = 10;
    const int perf_iters   = 100;

    // Define small helpers locally
    auto check_kernel_success = [&]() {
        CUDA_CHECK(cudaMemcpy(&h_status, d_status, sizeof(h_status), cudaMemcpyDeviceToHost));
        if (h_status != UCS_OK) {
            fprintf(stderr, "Rank %d kernel failed: %d\n", rank, h_status);
            MPI_Abort(MPI_COMM_WORLD, h_status);
        }
    };

    auto timed_kernel_iteration = [&]() -> float {
        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
        CUDA_CHECK(cudaEventRecord(ev_start));
        if (nvLink_transport) {
            do_alltoallv_kernel_nvLink_transport<<<kparams.num_blocks, kparams.num_threads>>>(kparams, d_ops, num_ops, d_status);
        } else {
            do_alltoallv_kernel_ib_transport<<<kparams.num_blocks, kparams.num_threads>>>(kparams, d_ops, num_ops, d_status);
        }
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float iter_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&iter_ms, ev_start, ev_stop));
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        return iter_ms;
    };

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        h_status = UCS_OK;
        CUDA_CHECK(cudaMemcpy(d_status, &h_status, sizeof(h_status), cudaMemcpyHostToDevice));
        float _ = timed_kernel_iteration();
        (void)_;
        check_kernel_success();
    }

    // Measure
    double sum_ms = 0.0, min_ms = 1e30, max_ms = 0.0;
    for (int i = 0; i < perf_iters; ++i) {
        h_status = UCS_OK;
        CUDA_CHECK(cudaMemcpy(d_status, &h_status, sizeof(h_status), cudaMemcpyHostToDevice));
        float iter_ms = timed_kernel_iteration();
        sum_ms += (double)iter_ms;
        if (iter_ms < min_ms) min_ms = iter_ms;
        if (iter_ms > max_ms) max_ms = iter_ms;
        check_kernel_success();
    }

    // Compute per-rank average latency (ms)
    double avg_ms = sum_ms / (double)perf_iters;

    // Gather only latency to rank 0 for summary
    std::vector<double> all_avg_ms;
    if (rank == 0) {
        all_avg_ms.resize(world_size);
    }
    MPI_CHECK(MPI_Gather(&avg_ms, 1, MPI_DOUBLE, rank == 0 ? all_avg_ms.data() : nullptr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        double lat_min = 1e30, lat_max = 0.0, lat_sum = 0.0;
        for (int r = 0; r < world_size; ++r) {
            lat_sum += all_avg_ms[r];
            if (all_avg_ms[r] < lat_min) lat_min = all_avg_ms[r];
            if (all_avg_ms[r] > lat_max) lat_max = all_avg_ms[r];
        }
        double lat_avg = lat_sum / (double)world_size;
        // Use rank 0 bytes to compute a simple throughput estimate
        unsigned long long rank0_bytes = 0ULL;
        for (int p = 0; p < world_size; ++p) {
            if (p == 0) continue;
            rank0_bytes += (unsigned long long)sendcounts[p];
        }
        double agg_gbps = (double)rank0_bytes / (lat_avg / 1000.0) / 1e9;
        printf("UCX device kernel perf (iters=%d, threads=%u, blocks=%u, ops=%u)\n",
               perf_iters, kparams.num_threads, kparams.num_blocks, num_ops);
        printf("  Kernel latency (ms): avg=%.3f min=%.3f max=%.3f\n", lat_avg, lat_min, lat_max);
        printf("  Throughput estimate using rank0 bytes (GB/s): %.3f\n", agg_gbps);
        fflush(stdout);
    }
}


int main(int argc, char **argv)
{
    // MPI Init
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) {
        if (rank == 0) fprintf(stderr, "Run with at least 2 ranks.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Get local rank
    int local_rank = 0, local_size = 0;
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_size(local_comm, &local_size));
    MPI_CHECK(MPI_Comm_free(&local_comm));

    // CUDA Init
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        fprintf(stderr, "No CUDA devices available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (ndev < local_size) {
        fprintf(stderr, "Not enough CUDA devices available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int dev = local_rank;
    CUDA_CHECK(cudaSetDevice(dev));
    // Ensure primary context is created and current for UCX driver API hooks
    CUDA_CHECK(cudaFree(0));

    // Init UCP
    ucp_context_h ucp_ctx = nullptr; ucp_worker_h worker = nullptr;
    init_ucp(ucp_ctx, worker);

    // Create endpoints to all peers
    std::vector<ucp_ep_h> eps; eps.reserve(world_size);
    create_all_endpoints(worker, rank, world_size, eps);

    // Set sendcounts/displs (with variable sizes)
    size_t base_len = 1 << 20;
    std::vector<int> sendcounts(world_size, 0), senddispls(world_size, 0);
    for (int dst = 0; dst < world_size; ++dst) {
        size_t len = base_len + (((rank * world_size + dst) % 4) * 256);
        assert(len <= INT_MAX);
        sendcounts[dst] = len;
    }
    for (int i = 1; i < world_size; ++i) senddispls[i] = senddispls[i-1] + sendcounts[i-1];
    size_t total_send = senddispls.back() + sendcounts.back();

    // Compute recvcounts via alltoall
    std::vector<int> recvcounts(world_size, 0), recvdispls(world_size, 0);
    MPI_CHECK(MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD));
    for (int i = 1; i < world_size; ++i) recvdispls[i] = recvdispls[i-1] + recvcounts[i-1];
    size_t total_recv = recvdispls.back() + recvcounts.back();

    // Allocate CUDA send/recv buffers
    void *send_buf = nullptr; void *recv_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&send_buf, total_send));
    CUDA_CHECK(cudaMalloc(&recv_buf, total_recv));

    // Fill send segments with a dst-varying byte; clear recv
    for (int dst = 0; dst < world_size; ++dst) {
        void *seg = (void*)((uintptr_t)send_buf + senddispls[dst]);
        unsigned char dst_byte = (unsigned char)(0x10 + ((rank + dst) & 0xEF));
        CUDA_CHECK(cudaMemset(seg, dst_byte, sendcounts[dst]));
    }
    CUDA_CHECK(cudaMemset(recv_buf, 0x00, total_recv));

    // Register recv memory and pack rkey
    ucp_mem_map_params_t mmap_params; memset(&mmap_params, 0, sizeof(mmap_params));
    mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH | UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    mmap_params.address    = recv_buf;
    mmap_params.length     = total_recv;
    mmap_params.memory_type= UCS_MEMORY_TYPE_CUDA;

    ucp_mem_h recv_memh = nullptr;
    UCP_CHECK(ucp_mem_map(ucp_ctx, &mmap_params, &recv_memh), "ucp_mem_map(recv)");

    void *rkey_buf = nullptr; size_t rkey_size = 0;
    UCP_CHECK(ucp_rkey_pack(ucp_ctx, recv_memh, &rkey_buf, &rkey_size), "ucp_rkey_pack");

    // Share base remote address and rkey to all peers (variable sizes)
    uint64_t my_remote_addr = (uint64_t)recv_buf;
    struct { uint64_t addr; uint32_t size; } header { my_remote_addr, (uint32_t)rkey_size };

    std::vector<uint8_t> my_blob(sizeof(header) + rkey_size);
    memcpy(my_blob.data(), &header, sizeof(header));
    memcpy(my_blob.data() + sizeof(header), rkey_buf, rkey_size);

    std::vector<int> blob_sizes(world_size, 0), blob_displs(world_size, 0);
    MPI_CHECK(MPI_Allgather(&(header.size), 1, MPI_INT, blob_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD));
    for (int i = 0; i < world_size; ++i) blob_sizes[i] += sizeof(header);
    for (int i = 1; i < world_size; ++i) blob_displs[i] = blob_displs[i-1] + blob_sizes[i-1];
    int total_blob = 0; for (int s : blob_sizes) total_blob += s;
    std::vector<uint8_t> all_blob(total_blob);
    MPI_CHECK(MPI_Allgatherv(my_blob.data(), my_blob.size(), MPI_BYTE,
                             all_blob.data(), blob_sizes.data(), blob_displs.data(), MPI_BYTE, MPI_COMM_WORLD));

    // Unpack each peer's rkey and record remote base address
    std::vector<ucp_rkey_h> peer_rkeys(world_size, nullptr);
    std::vector<uint64_t>   peer_bases(world_size, 0);
    for (int p = 0; p < world_size; ++p) {
        const uint8_t *ptr = all_blob.data() + blob_displs[p];
        struct { uint64_t addr; uint32_t size; } ph;
        memcpy(&ph, ptr, sizeof(ph));
        peer_bases[p] = ph.addr;
        const void *prkey = ptr + sizeof(ph);
        if (p != rank) {
            UCP_CHECK(ucp_ep_rkey_unpack(eps[p], prkey, &peer_rkeys[p]), "ucp_ep_rkey_unpack");
        }
    }

    // Gather each rank's recvdispls so senders can compute remote offsets
    std::vector<int> all_recvdispls(world_size * world_size, 0);
    MPI_CHECK(MPI_Allgather(recvdispls.data(), world_size, MPI_INT,
                            all_recvdispls.data(), world_size, MPI_INT, MPI_COMM_WORLD));

    // Map local send buffer for device-side API
    ucp_mem_h send_memh = nullptr;
    ucp_mem_map_params_t mmap_send; memset(&mmap_send, 0, sizeof(mmap_send));
    mmap_send.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH | UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    mmap_send.address    = send_buf;
    mmap_send.length     = total_send;
    mmap_send.memory_type= UCS_MEMORY_TYPE_CUDA;
    UCP_CHECK(ucp_mem_map(ucp_ctx, &mmap_send, &send_memh), "ucp_mem_map(send)");

    // Create one device mem list per peer (excluding self), each with a single element
    std::vector<ucp_device_mem_list_handle_h> mem_lists(world_size, nullptr);
    std::vector<unsigned> element_index(world_size, 0);
    for (int p = 0; p < world_size; ++p) {
        if (p == rank) continue;
        ucp_device_mem_list_elem_t elem;
        elem.field_mask = UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH |
                          UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY |
                          UCP_DEVICE_MEM_LIST_ELEM_FIELD_LOCAL_ADDR |
                          UCP_DEVICE_MEM_LIST_ELEM_FIELD_REMOTE_ADDR |
                          UCP_DEVICE_MEM_LIST_ELEM_FIELD_LENGTH;
        elem.memh        = send_memh;
        elem.rkey        = peer_rkeys[p];
        elem.local_addr  = send_buf;
        elem.length      = total_send;
        elem.remote_addr = peer_bases[p];

        ucp_device_mem_list_params_t ml_params;
        memset(&ml_params, 0, sizeof(ml_params));
        ml_params.field_mask   = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
                                 UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE |
                                 UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS;
        ml_params.element_size = sizeof(elem);
        ml_params.num_elements = 1;
        ml_params.elements     = &elem;

        ucs_status_t st;
        do {
            st = ucp_device_mem_list_create(eps[p], &ml_params, &mem_lists[p]);
            if (st == UCS_ERR_NOT_CONNECTED) {
                ucp_worker_progress(worker);
            }
        } while (st == UCS_ERR_NOT_CONNECTED);
        UCP_CHECK(st, "ucp_device_mem_list_create(per-peer)");
        element_index[p] = 0; // single element
    }

    // Build PUT operations: for each peer, compute local and remote offsets
    std::vector<put_op_t> ops; ops.reserve(world_size);
    for (int p = 0; p < world_size; ++p) {
        if (p == rank) continue;
        put_op_t op;
        op.list_handle_index = p;
        op.element_index     = element_index[p];
        op.local_offset   = (size_t)senddispls[p];
        size_t remote_off = all_recvdispls[p * world_size + rank];
        op.remote_offset  = remote_off;
        op.length         = sendcounts[p];
        ops.push_back(op);
    }

    // Upload ops to device
    put_op_t *d_ops = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ops, ops.size() * sizeof(put_op_t)));
    CUDA_CHECK(cudaMemcpy(d_ops, ops.data(), ops.size() * sizeof(put_op_t), cudaMemcpyHostToDevice));

    // Prepare kernel params
    kernel_params_t kparams = {};
    const unsigned num_ops = static_cast<unsigned>(ops.size());
    unsigned threads_per_block = 0;
    unsigned num_blocks = 0;
    if (nvLink_transport) {
        // One block per peer, all threads in block cooperate on the same PUT
        threads_per_block = 128; // many threads to drive nvLink
        num_blocks        = (num_ops > 0) ? num_ops : 1;
        kparams.level     = UCS_DEVICE_LEVEL_BLOCK;
    } else {
        // Single block, one thread per peer
        num_blocks        = 1;
        threads_per_block = (num_ops > 0) ? num_ops : 1;
        kparams.level     = UCS_DEVICE_LEVEL_THREAD;
    }
    kparams.num_threads  = threads_per_block;
    kparams.num_blocks   = num_blocks;
    kparams.with_request = false;
    // Upload mem list handle array to device
    ucp_device_mem_list_handle_h *d_mem_lists = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mem_lists, sizeof(ucp_device_mem_list_handle_h) * world_size));
    CUDA_CHECK(cudaMemcpy(d_mem_lists, mem_lists.data(), sizeof(ucp_device_mem_list_handle_h) * world_size, cudaMemcpyHostToDevice));
    kparams.mem_lists  = d_mem_lists;
    kparams.num_lists  = world_size;

    // Launch kernel
    ucs_status_t *d_status = nullptr; ucs_status_t h_status = UCS_OK;
    CUDA_CHECK(cudaMalloc(&d_status, sizeof(*d_status)));
    CUDA_CHECK(cudaMemcpy(d_status, &h_status, sizeof(h_status), cudaMemcpyHostToDevice));

    if (nvLink_transport) {
        do_alltoallv_kernel_nvLink_transport<<<kparams.num_blocks, kparams.num_threads>>>(kparams, d_ops, num_ops, d_status);
    } else {
        do_alltoallv_kernel_ib_transport<<<kparams.num_blocks, kparams.num_threads>>>(kparams, d_ops, num_ops, d_status);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_status, d_status, sizeof(h_status), cudaMemcpyDeviceToHost));
    if (h_status != UCS_OK) {
        fprintf(stderr, "Rank %d kernel failed: %d\n", rank, h_status);
        MPI_Abort(MPI_COMM_WORLD, h_status);
    }

    // Handle self-copy on host to emulate alltoallv semantics for self
    void *dst = (void*)((uintptr_t)recv_buf + recvdispls[rank]);
    const void *src = (const void*)((uintptr_t)send_buf + senddispls[rank]);
    CUDA_CHECK(cudaMemcpy(dst, src, sendcounts[rank], cudaMemcpyDeviceToDevice));

    // Validate
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    for (int s = 0; s < world_size; ++s) {
        size_t len = recvcounts[s];
        std::vector<uint8_t> host_check(len, 0);
        const void *seg = (const void*)((uintptr_t)recv_buf + recvdispls[s]);
        CUDA_CHECK(cudaMemcpy(host_check.data(), seg, len, cudaMemcpyDeviceToHost));
        unsigned char expected = (unsigned char)(0x10 + ((s + rank) & 0xEF));
        for (size_t i = 0; i < len; ++i) {
            if (host_check[i] != expected) {
                fprintf(stderr, "Rank %d validation failed for segment from %d (expected 0x%02x)\n", rank, s, expected);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // Benchmark
    run_device_kernel_benchmark(rank, world_size, kparams, d_ops, num_ops, d_status, h_status, sendcounts);

    // Cleanup
    CUDA_CHECK(cudaFree(d_ops));
    CUDA_CHECK(cudaFree(d_mem_lists));
    CUDA_CHECK(cudaFree(d_status));
    for (int p = 0; p < world_size; ++p) {
        if (p == rank) continue;
        if (mem_lists[p]) ucp_device_mem_list_release(mem_lists[p]);
    }
    for (int p = 0; p < world_size; ++p) {
        if (peer_rkeys[p]) ucp_rkey_destroy(peer_rkeys[p]);
        if (p != rank && eps[p]) ucp_ep_destroy(eps[p]);
    }
    ucp_rkey_buffer_release(rkey_buf);
    UCP_CHECK(ucp_mem_unmap(ucp_ctx, send_memh), "ucp_mem_unmap(send)");
    UCP_CHECK(ucp_mem_unmap(ucp_ctx, recv_memh), "ucp_mem_unmap(recv)");
    ucp_worker_destroy(worker);
    ucp_cleanup(ucp_ctx);

    CUDA_CHECK(cudaFree(send_buf));
    CUDA_CHECK(cudaFree(recv_buf));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    MPI_Finalize();

    return 0;
}
