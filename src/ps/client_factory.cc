#include "ps/client_factory.h"

#include <stdexcept>

#include "base/factory.h"
#include "ps/grpc/grpc_ps_client.h"
#include "ps/local_shm/local_shm_client.h"
#include "ps/rdma/rdma_ps_client_adapter.h"

#ifdef RECSTORE_HAS_BRPC_PS_CLIENT
#  include "ps/brpc/brpc_ps_client.h"
#endif

namespace recstore {

namespace {

const char* TypeKeyForFactory(PSClientType type) {
  switch (type) {
  case PSClientType::kGrpc:
    return "grpc";
  case PSClientType::kBrpc:
    return "brpc";
  case PSClientType::kLocalShm:
    return "local_shm";
  case PSClientType::kRdma:
    return "rdma";
  }

  throw std::invalid_argument("Unknown PSClientType");
}

} // namespace

std::unique_ptr<BasePSClient>
CreatePSClient(const PSClientCreateOptions& options) {
  if (options.type == PSClientType::kRdma) {
    return std::make_unique<RDMAPSClientAdapter>(options.raw_config);
  }

  auto& creators = base::Factory<BasePSClient, json>::creators();
  auto creator   = creators.find(TypeKeyForFactory(options.type));
  if (creator != creators.end() && creator->second != nullptr) {
    return std::unique_ptr<BasePSClient>(
        creator->second->create(options.transport_config));
  }

  if (options.type == PSClientType::kGrpc) {
    return std::make_unique<GRPCParameterClient>(options.transport_config);
  }

  if (options.type == PSClientType::kBrpc) {
#ifdef RECSTORE_HAS_BRPC_PS_CLIENT
    return std::make_unique<BRPCParameterClient>(options.transport_config);
#else
    throw std::runtime_error("bRPC parameter client is not available");
#endif
  }

  if (options.type == PSClientType::kLocalShm) {
    return std::make_unique<LocalShmPSClient>(options.transport_config);
  }

  throw std::runtime_error("Failed to create PS client");
}

} // namespace recstore
