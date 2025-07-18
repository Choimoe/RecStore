#include "ucc_context.h"
#include <cstdlib>

namespace recstore {
namespace framework {

UCCContext& UCCContext::GetInstance() {
    static UCCContext instance;
    return instance;
}

UCCContext::UCCContext() {
    ucc_lib_config_h lib_config;
    check_ucc_status(ucc_lib_config_read(NULL, NULL, &lib_config), "ucc_lib_config_read");
    
    ucc_lib_params_t lib_params = {};
    lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = UCC_THREAD_MULTIPLE;
    check_ucc_status(ucc_init(&lib_params, lib_config, &lib_), "ucc_init");
    ucc_lib_config_release(lib_config);

    ucc_context_config_h context_config;
    check_ucc_status(ucc_context_config_read(lib_, NULL, &context_config), "ucc_context_config_read");
    
    ucc_context_params_t context_params = {};
    context_params.mask = UCC_CONTEXT_PARAM_FIELD_TYPE | UCC_CONTEXT_PARAM_FIELD_OOB;
    context_params.type = UCC_CONTEXT_SHARED;
    
    const char* rank_str = std::getenv("OMPI_COMM_WORLD_RANK");
    const char* size_str = std::getenv("OMPI_COMM_WORLD_SIZE");
    if (!rank_str || !size_str) {
        throw std::runtime_error("UCC requires MPI environment variables (e.g., OMPI_COMM_WORLD_RANK/SIZE)");
    }
    rank_ = std::stoi(rank_str);
    world_size_ = std::stoi(size_str);

    check_ucc_status(ucc_context_create(lib_, &context_params, context_config, &context_), "ucc_context_create");
    ucc_context_config_release(context_config);

    ucc_team_params_t team_params;
    team_params.mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE | UCC_TEAM_PARAM_FIELD_OOB;
    team_params.oob.allgather = nullptr;
    team_params.oob.req_test = nullptr;
    team_params.oob.req_free = nullptr;
    team_params.oob.n_oob_eps = world_size_;
    team_params.oob.oob_ep = rank_;
    team_params.ep = rank_;
    team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
    check_ucc_status(ucc_team_create_post(&context_, 1, &team_params, &team_), "ucc_team_create_post");
    
    ucc_status_t status;
    while (UCC_INPROGRESS == (status = ucc_team_create_test(team_))) {}
    check_ucc_status(status, "ucc_team_create_test");

    std::cout << "UCCContext initialized for rank " << rank_ << " of " << world_size_ << std::endl;
}

UCCContext::~UCCContext() {
    if (team_) {
        ucc_team_destroy(team_);
    }
    if (context_) {
        ucc_context_destroy(context_);
    }
    if (lib_) {
        ucc_finalize(lib_);
    }
    std::cout << "UCCContext destroyed for rank " << rank_ << std::endl;
}

void UCCContext::check_ucc_status(ucc_status_t status, const std::string& message) {
    if (status != UCC_OK) {
        throw std::runtime_error("UCC Error: " + message + " failed with status " + std::to_string(status));
    }
}

void UCCContext::Barrier() {
    ucc_coll_req_h request;
    ucc_coll_args_t args = {};
    args.mask = 0;
    args.coll_type = UCC_COLL_TYPE_BARRIER;
    
    check_ucc_status(ucc_collective_init(&args, &request, team_), "ucc_collective_init for barrier");
    check_ucc_status(ucc_collective_post(request), "ucc_collective_post for barrier");
    
    ucc_status_t status;
    while (UCC_INPROGRESS == (status = ucc_collective_test(request))) {}
    ucc_collective_finalize(request);
    check_ucc_status(status, "ucc_collective_test for barrier");
}

std::vector<char> UCCContext::Allgatherv(const std::vector<char>& send_buffer) {
    // 1. Allgather 发送缓冲区的大小
    std::vector<int> recv_counts(world_size_);
    int send_count = send_buffer.size();

    ucc_coll_req_h req;
    ucc_coll_args_t args = {};
    args.mask = 0;
    args.coll_type = UCC_COLL_TYPE_ALLGATHER;
    args.src.info.buffer = &send_count;
    args.src.info.count = 1;
    args.src.info.datatype = UCC_DT_INT32;
    args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    args.dst.info.buffer = recv_counts.data();
    args.dst.info.count = 1; // 每个进程接收1个int
    args.dst.info.datatype = UCC_DT_INT32;
    args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

    check_ucc_status(ucc_collective_init(&args, &req, team_), "ucc_collective_init for allgather (counts)");
    check_ucc_status(ucc_collective_post(req), "ucc_collective_post for allgather (counts)");
    ucc_status_t status;
    while (UCC_INPROGRESS == (status = ucc_collective_test(req))) {}
    ucc_collective_finalize(req);
    check_ucc_status(status, "ucc_collective_test for allgather (counts)");

    std::vector<int> displacements(world_size_, 0);
    int total_size = recv_counts[0];
    for (int i = 1; i < world_size_; ++i) {
        displacements[i] = displacements[i-1] + recv_counts[i-1];
        total_size += recv_counts[i];
    }
    
    std::vector<char> recv_buffer(total_size);

    args = {};
    args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
    args.flags = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER | UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER;
    args.coll_type = UCC_COLL_TYPE_ALLGATHERV;
    args.src.info.buffer = (void*)send_buffer.data();
    args.src.info.count = send_buffer.size();
    args.src.info.datatype = UCC_DT_INT8;
    args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    args.dst.info_v.buffer = recv_buffer.data();
    args.dst.info_v.counts = (ucc_count_t*)recv_counts.data();
    args.dst.info_v.displacements = (ucc_aint_t*)displacements.data();
    args.dst.info_v.datatype = UCC_DT_INT8;
    args.dst.info_v.mem_type = UCC_MEMORY_TYPE_HOST;

    check_ucc_status(ucc_collective_init(&args, &req, team_), "ucc_collective_init for allgatherv");
    check_ucc_status(ucc_collective_post(req), "ucc_collective_post for allgatherv");
    while (UCC_INPROGRESS == (status = ucc_collective_test(req))) {}
    ucc_collective_finalize(req);
    check_ucc_status(status, "ucc_collective_test for allgatherv");
    
    return recv_buffer;
}

} // namespace framework
} // namespace recstore
