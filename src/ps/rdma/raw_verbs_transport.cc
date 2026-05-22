#include "ps/rdma/raw_verbs_transport.h"

#include <arpa/inet.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "ps/base/Postoffice.h"

namespace petps {
namespace {

constexpr int kRawVerbsPort          = 1;
constexpr int kRawVerbsGidIndex      = 1;
constexpr std::uint32_t kRawVerbsPsn = 3185;
constexpr int kRawVerbsCqDepth       = 4096;
constexpr int kRawVerbsRecvDepth     = 1024;

std::string IbvError(const char* op) { return std::string(op) + " failed"; }

ibv_context* OpenDeviceForNuma(int numa_id) {
  int device_count     = 0;
  ibv_device** devices = ibv_get_device_list(&device_count);
  if (devices == nullptr || device_count == 0) {
    throw std::runtime_error("no RDMA devices found");
  }
  const int device_index = SelectRawVerbsDeviceIndex(numa_id, device_count);
  ibv_context* context   = ibv_open_device(devices[device_index]);
  ibv_free_device_list(devices);
  if (context == nullptr) {
    throw std::runtime_error("ibv_open_device failed");
  }
  return context;
}

void ModifyQpToInit(ibv_qp* qp) {
  ibv_qp_attr attr{};
  attr.qp_state        = IBV_QPS_INIT;
  attr.port_num        = kRawVerbsPort;
  attr.pkey_index      = 0;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE |
                         IBV_ACCESS_REMOTE_ATOMIC;
  const int flags =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  if (ibv_modify_qp(qp, &attr, flags) != 0) {
    throw std::runtime_error(IbvError("ibv_modify_qp INIT"));
  }
}

void FillAhAttr(ibv_ah_attr* ah_attr,
                std::uint16_t remote_lid,
                const std::uint8_t* remote_gid) {
  std::memset(ah_attr, 0, sizeof(*ah_attr));
  ah_attr->dlid          = remote_lid;
  ah_attr->sl            = 0;
  ah_attr->src_path_bits = 0;
  ah_attr->port_num      = kRawVerbsPort;
  if (remote_gid != nullptr) {
    ah_attr->is_global = 1;
    std::memcpy(&ah_attr->grh.dgid, remote_gid, 16);
    ah_attr->grh.sgid_index = kRawVerbsGidIndex;
    ah_attr->grh.hop_limit  = 1;
  }
}

void ModifyQpToRtr(ibv_qp* qp, const RawVerbsNodeMeta& remote) {
  ibv_qp_attr attr{};
  attr.qp_state           = IBV_QPS_RTR;
  attr.path_mtu           = IBV_MTU_4096;
  attr.dest_qp_num        = remote.qpn;
  attr.rq_psn             = remote.psn;
  attr.max_dest_rd_atomic = 16;
  attr.min_rnr_timer      = 12;
  FillAhAttr(&attr.ah_attr, remote.lid, remote.gid);
  const int flags =
      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
      IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  if (ibv_modify_qp(qp, &attr, flags) != 0) {
    throw std::runtime_error(IbvError("ibv_modify_qp RTR"));
  }
}

void ModifyQpToRts(ibv_qp* qp) {
  ibv_qp_attr attr{};
  attr.qp_state      = IBV_QPS_RTS;
  attr.sq_psn        = kRawVerbsPsn;
  attr.timeout       = 14;
  attr.retry_cnt     = 7;
  attr.rnr_retry     = 7;
  attr.max_rd_atomic = 16;
  const int flags =
      IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
      IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
  if (ibv_modify_qp(qp, &attr, flags) != 0) {
    throw std::runtime_error(IbvError("ibv_modify_qp RTS"));
  }
}

std::string HexEncodeBytes(const void* data, std::size_t size) {
  static constexpr char kHex[] = "0123456789abcdef";
  const auto* bytes            = static_cast<const std::uint8_t*>(data);
  std::string out(size * 2, '\0');
  for (std::size_t i = 0; i < size; ++i) {
    out[i * 2]     = kHex[bytes[i] >> 4];
    out[i * 2 + 1] = kHex[bytes[i] & 0x0f];
  }
  return out;
}

int HexValue(char ch) {
  if (ch >= '0' && ch <= '9') {
    return ch - '0';
  }
  if (ch >= 'a' && ch <= 'f') {
    return 10 + ch - 'a';
  }
  if (ch >= 'A' && ch <= 'F') {
    return 10 + ch - 'A';
  }
  return -1;
}

bool DecodeRawVerbsNodeMeta(const std::string& value, RawVerbsNodeMeta* meta) {
  if (meta == nullptr) {
    return false;
  }
  if (value.size() == sizeof(RawVerbsNodeMeta)) {
    std::memcpy(meta, value.data(), sizeof(RawVerbsNodeMeta));
    return true;
  }
  if (value.size() != sizeof(RawVerbsNodeMeta) * 2) {
    return false;
  }
  auto* bytes = reinterpret_cast<std::uint8_t*>(meta);
  for (std::size_t i = 0; i < sizeof(RawVerbsNodeMeta); ++i) {
    const int high = HexValue(value[i * 2]);
    const int low  = HexValue(value[i * 2 + 1]);
    if (high < 0 || low < 0) {
      return false;
    }
    bytes[i] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return true;
}

} // namespace

struct RawVerbsTransport::Impl {
  explicit Impl(const RawVerbsConfig& c)
      : config(c), allocator(c.local_region_bytes, c.allocation_start_offset) {}

  RawVerbsConfig config;
  ibv_context* context    = nullptr;
  ibv_pd* pd              = nullptr;
  ibv_cq* cq              = nullptr;
  ibv_mr* local_mr        = nullptr;
  void* local_base        = nullptr;
  bool owns_local_base    = false;
  std::size_t local_bytes = 0;
  RawVerbsRegionAllocator allocator;
  std::vector<RawVerbsNodeMeta> metas;
  std::vector<RawVerbsRemoteMemory> remotes;
  std::vector<ibv_qp*> qps;
  ibv_wc wc_batch[kRawVerbsPollBatchSize] = {};
  RawVerbsCompletionBatchCursor batch_cursor;
};

RawVerbsTransport::RawVerbsTransport(const RawVerbsConfig& config)
    : impl_(std::make_unique<Impl>(config)) {
  impl_->context = OpenDeviceForNuma(config.numa_id);
  impl_->pd      = ibv_alloc_pd(impl_->context);
  if (impl_->pd == nullptr) {
    throw std::runtime_error("ibv_alloc_pd failed");
  }
  impl_->cq =
      ibv_create_cq(impl_->context, kRawVerbsCqDepth, nullptr, nullptr, 0);
  if (impl_->cq == nullptr) {
    throw std::runtime_error("ibv_create_cq failed");
  }

  impl_->local_bytes = config.local_region_bytes;
  impl_->local_base  = reinterpret_cast<void*>(config.local_base_addr);
  if (impl_->local_base == nullptr) {
    const int rc = posix_memalign(&impl_->local_base, 4096, impl_->local_bytes);
    if (rc != 0) {
      throw std::runtime_error("posix_memalign failed for raw verbs region");
    }
    impl_->owns_local_base = true;
  }
  impl_->local_mr = ibv_reg_mr(
      impl_->pd,
      impl_->local_base,
      impl_->local_bytes,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
          IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
  if (impl_->local_mr == nullptr) {
    throw std::runtime_error("ibv_reg_mr failed");
  }
  if (config.reserved_region_bytes != 0) {
    impl_->allocator.SetReservedRegion(
        {config.reserved_region_offset, config.reserved_region_bytes});
  }

  const int node_count = config.num_servers + config.num_clients;
  impl_->qps.resize(static_cast<std::size_t>(node_count), nullptr);
  for (int node = 0; node < node_count; ++node) {
    if (!ShouldRawVerbsConnectToNode(config, node)) {
      continue;
    }
    ibv_qp_init_attr init_attr{};
    init_attr.send_cq             = impl_->cq;
    init_attr.recv_cq             = impl_->cq;
    init_attr.qp_type             = IBV_QPT_RC;
    init_attr.cap.max_send_wr     = 1024;
    init_attr.cap.max_recv_wr     = kRawVerbsRecvDepth;
    init_attr.cap.max_send_sge    = 1;
    init_attr.cap.max_recv_sge    = 1;
    init_attr.cap.max_inline_data = 64;
    ibv_qp* qp                    = ibv_create_qp(impl_->pd, &init_attr);
    if (qp == nullptr) {
      throw std::runtime_error("ibv_create_qp failed");
    }
    impl_->qps[static_cast<std::size_t>(node)] = qp;
  }
}

RawVerbsTransport::~RawVerbsTransport() {
  if (!impl_) {
    return;
  }
  for (ibv_qp* qp : impl_->qps) {
    if (qp != nullptr) {
      ibv_destroy_qp(qp);
    }
  }
  if (impl_->local_mr != nullptr) {
    ibv_dereg_mr(impl_->local_mr);
  }
  if (impl_->cq != nullptr) {
    ibv_destroy_cq(impl_->cq);
  }
  if (impl_->pd != nullptr) {
    ibv_dealloc_pd(impl_->pd);
  }
  if (impl_->context != nullptr) {
    ibv_close_device(impl_->context);
  }
  if (impl_->owns_local_base && impl_->local_base != nullptr) {
    free(impl_->local_base);
  }
}

void RawVerbsTransport::RegisterThread() {}

void* RawVerbsTransport::AllocateRegistered(std::size_t bytes) {
  const std::uint64_t offset = impl_->allocator.Allocate(bytes);
  return static_cast<char*>(impl_->local_base) + offset;
}

std::uint64_t RawVerbsTransport::SaveAllocationState() const {
  return impl_->allocator.Checkpoint();
}

void RawVerbsTransport::RestoreAllocationState(std::uint64_t checkpoint) {
  impl_->allocator.Restore(checkpoint);
}

GlobalAddress RawVerbsTransport::LocalAddress(void* ptr) const {
  const auto base = reinterpret_cast<std::uintptr_t>(impl_->local_base);
  const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
  if (addr < base || addr >= base + impl_->local_bytes) {
    throw std::runtime_error("pointer outside raw verbs local region");
  }
  return GlobalAddress{
      static_cast<std::uint16_t>(impl_->config.global_id),
      static_cast<std::uint64_t>(addr - base),
  };
}

void* RawVerbsTransport::LocalPointer(GlobalAddress address) const {
  if (address.nodeID != impl_->config.global_id) {
    throw std::runtime_error(
        "raw verbs local pointer requested for remote node");
  }
  if (address.offset >= impl_->local_bytes) {
    throw std::runtime_error("raw verbs local offset outside region");
  }
  return static_cast<char*>(impl_->local_base) + address.offset;
}

RawVerbsNodeMeta RawVerbsTransport::LocalMeta() const {
  ibv_port_attr port_attr{};
  if (ibv_query_port(impl_->context, kRawVerbsPort, &port_attr) != 0) {
    throw std::runtime_error("ibv_query_port failed");
  }
  ibv_gid gid{};
  if (ibv_query_gid(impl_->context, kRawVerbsPort, kRawVerbsGidIndex, &gid) !=
      0) {
    throw std::runtime_error("ibv_query_gid failed");
  }
  RawVerbsNodeMeta meta{};
  meta.node_id   = static_cast<std::uint16_t>(impl_->config.global_id);
  meta.lid       = port_attr.lid;
  meta.psn       = kRawVerbsPsn;
  meta.rkey      = impl_->local_mr->rkey;
  meta.base_addr = reinterpret_cast<std::uint64_t>(impl_->local_base);
  std::memcpy(meta.gid, &gid, sizeof(meta.gid));
  return meta;
}

void RawVerbsTransport::PublishAndConnect() {
  const int node_count = impl_->config.num_servers + impl_->config.num_clients;
  const RawVerbsNodeMeta local = LocalMeta();
  for (int node = 0; node < node_count; ++node) {
    if (!ShouldRawVerbsConnectToNode(impl_->config, node)) {
      continue;
    }
    RawVerbsNodeMeta peer_local = local;
    peer_local.qpn = impl_->qps[static_cast<std::size_t>(node)]->qp_num;
    XPostoffice::GetInstance()->MemCachedSet(
        RawVerbsMetaKey(impl_->config.global_id,
                        impl_->config.local_lane,
                        node,
                        impl_->config.remote_lane),
        HexEncodeBytes(&peer_local, sizeof(peer_local)));
  }

  impl_->metas.assign(static_cast<std::size_t>(node_count), RawVerbsNodeMeta{});
  impl_->remotes.assign(
      static_cast<std::size_t>(node_count), RawVerbsRemoteMemory{});

  for (int node = 0; node < node_count; ++node) {
    if (node == impl_->config.global_id) {
      impl_->metas[static_cast<std::size_t>(node)]   = local;
      impl_->remotes[static_cast<std::size_t>(node)] = RawVerbsRemoteMemory{
          local.node_id,
          local.base_addr,
          local.rkey,
      };
      continue;
    }
    if (!ShouldRawVerbsConnectToNode(impl_->config, node)) {
      continue;
    }
    std::string value;
    while (!XPostoffice::GetInstance()->MemCachedTryGet(
        RawVerbsMetaKey(node,
                        impl_->config.remote_lane,
                        impl_->config.global_id,
                        impl_->config.local_lane),
        &value)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    RawVerbsNodeMeta meta{};
    if (!DecodeRawVerbsNodeMeta(value, &meta)) {
      throw std::runtime_error(
          "raw verbs metadata size mismatch for node " + std::to_string(node));
    }
    impl_->metas[static_cast<std::size_t>(node)]   = meta;
    impl_->remotes[static_cast<std::size_t>(node)] = RawVerbsRemoteMemory{
        meta.node_id,
        meta.base_addr,
        meta.rkey,
    };
  }

  for (int node = 0; node < node_count; ++node) {
    if (!ShouldRawVerbsConnectToNode(impl_->config, node)) {
      continue;
    }
    ibv_qp* qp = impl_->qps[static_cast<std::size_t>(node)];
    ModifyQpToInit(qp);
    ModifyQpToRtr(qp, impl_->metas[static_cast<std::size_t>(node)]);
    ModifyQpToRts(qp);
    for (int i = 0; i < kRawVerbsRecvDepth; ++i) {
      ibv_recv_wr wr{};
      ibv_recv_wr* bad_wr = nullptr;
      wr.wr_id            = static_cast<std::uint64_t>(node);
      wr.sg_list          = nullptr;
      wr.num_sge          = 0;
      if (ibv_post_recv(qp, &wr, &bad_wr) != 0) {
        throw std::runtime_error("ibv_post_recv failed");
      }
    }
  }
}

void RawVerbsTransport::Write(
    const void* local,
    GlobalAddress remote,
    std::size_t bytes,
    std::uint64_t wr_id,
    bool signaled) {
  if (remote.nodeID >= impl_->remotes.size()) {
    throw std::runtime_error("raw verbs write remote node out of range");
  }
  ibv_sge sge{};
  sge.addr   = reinterpret_cast<std::uint64_t>(local);
  sge.length = static_cast<std::uint32_t>(bytes);
  sge.lkey   = impl_->local_mr->lkey;
  ibv_send_wr wr{};
  wr.wr_id      = wr_id;
  wr.opcode     = IBV_WR_RDMA_WRITE;
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wr.sg_list    = &sge;
  wr.num_sge    = 1;
  wr.wr.rdma.remote_addr =
      impl_->remotes[remote.nodeID].base_addr + remote.offset;
  wr.wr.rdma.rkey     = impl_->remotes[remote.nodeID].rkey;
  ibv_send_wr* bad_wr = nullptr;
  if (ibv_post_send(impl_->qps[remote.nodeID], &wr, &bad_wr) != 0) {
    throw std::runtime_error("ibv_post_send write failed");
  }
}

void RawVerbsTransport::WriteWithImm(
    const void* local,
    GlobalAddress remote,
    std::size_t bytes,
    std::uint32_t imm_data,
    std::uint64_t wr_id,
    bool signaled) {
  if (remote.nodeID >= impl_->remotes.size()) {
    throw std::runtime_error(
        "raw verbs write-with-imm remote node out of range");
  }
  ibv_sge sge{};
  sge.addr   = reinterpret_cast<std::uint64_t>(local);
  sge.length = static_cast<std::uint32_t>(bytes);
  sge.lkey   = impl_->local_mr->lkey;
  ibv_send_wr wr{};
  wr.wr_id      = wr_id;
  wr.opcode     = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.imm_data   = htonl(imm_data);
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wr.sg_list    = &sge;
  wr.num_sge    = 1;
  wr.wr.rdma.remote_addr =
      impl_->remotes[remote.nodeID].base_addr + remote.offset;
  wr.wr.rdma.rkey     = impl_->remotes[remote.nodeID].rkey;
  ibv_send_wr* bad_wr = nullptr;
  if (ibv_post_send(impl_->qps[remote.nodeID], &wr, &bad_wr) != 0) {
    throw std::runtime_error("ibv_post_send write-with-imm failed");
  }
}

void RawVerbsTransport::Read(
    void* local,
    GlobalAddress remote,
    std::size_t bytes,
    std::uint64_t wr_id,
    bool signaled) {
  if (remote.nodeID >= impl_->remotes.size()) {
    throw std::runtime_error("raw verbs read remote node out of range");
  }
  ibv_sge sge{};
  sge.addr   = reinterpret_cast<std::uint64_t>(local);
  sge.length = static_cast<std::uint32_t>(bytes);
  sge.lkey   = impl_->local_mr->lkey;
  ibv_send_wr wr{};
  wr.wr_id      = wr_id;
  wr.opcode     = IBV_WR_RDMA_READ;
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wr.sg_list    = &sge;
  wr.num_sge    = 1;
  wr.wr.rdma.remote_addr =
      impl_->remotes[remote.nodeID].base_addr + remote.offset;
  wr.wr.rdma.rkey     = impl_->remotes[remote.nodeID].rkey;
  ibv_send_wr* bad_wr = nullptr;
  if (ibv_post_send(impl_->qps[remote.nodeID], &wr, &bad_wr) != 0) {
    throw std::runtime_error("ibv_post_send read failed");
  }
}

void RawVerbsTransport::SendDoorbell(
    std::uint16_t node_id, std::uint32_t imm_data, std::uint64_t wr_id) {
  if (node_id >= impl_->qps.size()) {
    throw std::runtime_error("raw verbs doorbell remote node out of range");
  }
  ibv_send_wr wr{};
  wr.opcode           = IBV_WR_SEND_WITH_IMM;
  wr.imm_data         = htonl(imm_data);
  wr.send_flags       = IBV_SEND_SIGNALED;
  wr.wr_id            = wr_id;
  wr.sg_list          = nullptr;
  wr.num_sge          = 0;
  ibv_send_wr* bad_wr = nullptr;
  if (ibv_post_send(impl_->qps[node_id], &wr, &bad_wr) != 0) {
    throw std::runtime_error("ibv_post_send doorbell failed");
  }
}

bool RawVerbsTransport::Poll(RawVerbsCompletion* completion, int timeout_ms) {
  const auto deadline =
      timeout_ms > 0
          ? std::chrono::steady_clock::now() +
                std::chrono::milliseconds(timeout_ms)
          : std::chrono::steady_clock::time_point::max();
  while (true) {
    if (!impl_->batch_cursor.HasCachedCompletion()) {
      const int n =
          ibv_poll_cq(impl_->cq, kRawVerbsPollBatchSize, impl_->wc_batch);
      if (n < 0) {
        throw std::runtime_error("ibv_poll_cq failed");
      }
      if (n == 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
          return false;
        }
        std::this_thread::yield();
        continue;
      }
      impl_->batch_cursor.Reset(impl_->wc_batch, n);
    }
    ibv_wc& wc = *impl_->batch_cursor.TakeCachedCompletion();
    if (wc.status != IBV_WC_SUCCESS) {
      throw std::runtime_error(
          std::string("raw verbs CQ error: ") + ibv_wc_status_str(wc.status));
    }
    if (completion != nullptr) {
      completion->wr_id    = wc.wr_id;
      completion->opcode   = wc.opcode;
      completion->has_imm  = (wc.wc_flags & IBV_WC_WITH_IMM) != 0;
      completion->imm_data = completion->has_imm ? ntohl(wc.imm_data) : 0;
    }
    if (wc.opcode == IBV_WC_RECV || wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      const std::uint16_t node_id = static_cast<std::uint16_t>(wc.wr_id);
      ibv_recv_wr wr{};
      ibv_recv_wr* bad_wr = nullptr;
      wr.wr_id            = node_id;
      wr.sg_list          = nullptr;
      wr.num_sge          = 0;
      if (node_id < impl_->qps.size() &&
          ibv_post_recv(impl_->qps[node_id], &wr, &bad_wr) != 0) {
        throw std::runtime_error("ibv_post_recv repost failed");
      }
    }
    return true;
  }
}

} // namespace petps
