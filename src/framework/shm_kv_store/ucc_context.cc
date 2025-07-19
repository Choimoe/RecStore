#include "ucc_context.h"
#include <cstdlib>
#include <chrono>
#include <thread>

// UCC OOB callbacks using MPI - Fixed version with synchronous operations
static ucc_status_t oob_allgather(
    void* sbuf, void* rbuf, size_t msglen, void* coll_info, void** req) {
  // Use synchronous MPI_Allgather instead of asynchronous
  MPI_Allgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, MPI_COMM_WORLD);
  *req = nullptr; // No request needed for synchronous operation
  return UCC_OK;
}

static ucc_status_t oob_req_test(void* req) {
  // For synchronous operations, always return OK
  return UCC_OK;
}

static ucc_status_t oob_req_free(void* req) {
  // Nothing to free for synchronous operations
  return UCC_OK;
}

namespace recstore {
namespace framework {

UCCContext& UCCContext::GetInstance() {
  static UCCContext instance;
  return instance;
}

UCCContext::UCCContext() {
  // Initialize MPI (safe to call multiple times)
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    MPI_Init(NULL, NULL);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  std::cout << "Rank " << rank_ << ": Starting UCC initialization..." << std::endl;

  // UCC library setup
  ucc_lib_config_h lib_config;
  check_ucc_status(
      ucc_lib_config_read(NULL, NULL, &lib_config), "ucc_lib_config_read");
  ucc_lib_params_t lib_params = {};
  lib_params.mask             = UCC_LIB_PARAM_FIELD_THREAD_MODE;
  lib_params.thread_mode      = UCC_THREAD_MULTIPLE;
  check_ucc_status(ucc_init(&lib_params, lib_config, &lib_), "ucc_init");
  ucc_lib_config_release(lib_config);

  std::cout << "Rank " << rank_ << ": UCC library initialized successfully." << std::endl;

  // UCC context setup with OOB
  ucc_context_config_h context_config;
  check_ucc_status(ucc_context_config_read(lib_, NULL, &context_config),
                   "ucc_context_config_read");
  ucc_context_params_t context_params = {};
  context_params.mask =
      UCC_CONTEXT_PARAM_FIELD_TYPE | UCC_CONTEXT_PARAM_FIELD_OOB;
  context_params.type          = UCC_CONTEXT_SHARED;
  context_params.oob.n_oob_eps = world_size_;
  context_params.oob.oob_ep    = rank_;
  context_params.oob.allgather = oob_allgather;
  context_params.oob.req_test  = oob_req_test;
  context_params.oob.req_free  = oob_req_free;
  context_params.oob.coll_info = NULL;
  check_ucc_status(
      ucc_context_create(lib_, &context_params, context_config, &context_),
      "ucc_context_create");
  ucc_context_config_release(context_config);

  std::cout << "Rank " << rank_ << ": UCC context created successfully." << std::endl;

  // Note: We don't create UCC team anymore since we use direct MPI operations
  team_ = nullptr;
  std::cout << "Rank " << rank_ << ": Skipping UCC team creation (using direct MPI operations)." << std::endl;

  std::cout << "UCCContext initialized for rank " << rank_ << " of "
            << world_size_ << std::endl;
}

UCCContext::~UCCContext() {
  if (team_)
    ucc_team_destroy(team_);
  if (context_)
    ucc_context_destroy(context_);
  if (lib_)
    ucc_finalize(lib_);
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized) {
    MPI_Finalize();
  }
  std::cout << "UCCContext destroyed for rank " << rank_ << std::endl;
}

void UCCContext::check_ucc_status(ucc_status_t status,
                                  const std::string& message) {
  if (status != UCC_OK) {
    throw std::runtime_error("UCC Error: " + message + " failed with status " +
                             std::to_string(status));
  }
}

void UCCContext::Barrier() {
  // Use direct MPI barrier instead of UCC barrier to avoid deadlocks
  std::cout << "Rank " << rank_ << ": Using MPI barrier..." << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "Rank " << rank_ << ": MPI barrier completed." << std::endl;
}

std::vector<char> UCCContext::Allgatherv(const std::vector<char>& send_buffer) {
  // Use direct MPI operations instead of UCC to avoid deadlocks
  
  // 1. Allgather send buffer sizes using MPI
  last_recv_counts_.assign(world_size_, 0);
  int send_count = send_buffer.size();
  
  std::cout << "Rank " << rank_ << ": Allgatherv called with send_buffer.size()=" << send_buffer.size() << std::endl;
  
  // Debug: Print first few bytes of send_buffer if not empty
  if (!send_buffer.empty()) {
    std::cout << "Rank " << rank_ << ": send_buffer first 16 bytes: ";
    for (size_t i = 0; i < std::min(send_buffer.size(), size_t(16)); ++i) {
      printf("%02x ", (unsigned char)send_buffer[i]);
    }
    std::cout << std::endl;
  }
  
  std::cout << "Rank " << rank_ << ": Using MPI_Allgather for counts... (send_count=" << send_count << ")" << std::endl;
  MPI_Allgather(&send_count, 1, MPI_INT, last_recv_counts_.data(), 1, MPI_INT, MPI_COMM_WORLD);
  std::cout << "Rank " << rank_ << ": MPI_Allgather for counts completed." << std::endl;

  // 2. Prepare for Allgatherv
  std::vector<int> displacements(world_size_, 0);
  int total_size = 0;
  for (int i = 0; i < world_size_; ++i) {
    displacements[i] =
        (i > 0) ? (displacements[i - 1] + last_recv_counts_[i - 1]) : 0;
    total_size += last_recv_counts_[i];
  }

  std::cout << "Rank " << rank_ << ": Total size = " << total_size << ", counts = [";
  for (int i = 0; i < world_size_; ++i) {
    std::cout << last_recv_counts_[i] << (i < world_size_ - 1 ? ", " : "");
  }
  std::cout << "]" << std::endl;

  std::vector<char> recv_buffer(total_size);

  // 3. Use MPI_Allgatherv directly
  // Handle empty buffer case by providing a valid pointer
  const void* send_ptr;
  int send_size = send_buffer.size();
  
  if (send_buffer.empty()) {
    // For empty buffer, use a dummy pointer (recv_buffer.data() is always valid)
    send_ptr = recv_buffer.data();
    std::cout << "Rank " << rank_ << ": Empty buffer, using dummy pointer" << std::endl;
  } else {
    send_ptr = send_buffer.data();
    std::cout << "Rank " << rank_ << ": Non-empty buffer, size=" << send_size << std::endl;
  }
  
  std::cout << "Rank " << rank_ << ": Using MPI_Allgatherv for data... (send_size=" << send_size << ")" << std::endl;
  MPI_Allgatherv(
      const_cast<void*>(send_ptr), send_size, MPI_BYTE,
      recv_buffer.data(), last_recv_counts_.data(), displacements.data(), MPI_BYTE,
      MPI_COMM_WORLD);
  std::cout << "Rank " << rank_ << ": MPI_Allgatherv for data completed." << std::endl;

  return recv_buffer;
}

} // namespace framework
} // namespace recstore
