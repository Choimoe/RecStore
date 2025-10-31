#include <folly/executors/CPUThreadPoolExecutor.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <cstdint>
#include <future>
#include <string>
#include <vector>
#include <thread>

#include "base/array.h"
#include "base/base.h"
#include "base/timer.h"
#include "base_ps/base_ps_server.h"
#include "base_ps/cache_ps_impl.h"
#include "base_ps/parameters.h"
#include "base/flatc.h"
#include "ps.grpc.pb.h"
#include "ps.pb.h"
#include "recstore_config.h"
#include <gperftools/profiler.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using recstoreps::CommandRequest;
using recstoreps::CommandResponse;
using recstoreps::GetParameterRequest;
using recstoreps::GetParameterResponse;
using recstoreps::PSCommand;
using recstoreps::PutParameterRequest;
using recstoreps::PutParameterResponse;

DEFINE_string(config_path, RECSTORE_PATH "/recstore_config.json",
              "config file path");

class ParameterServiceImpl final
    : public recstoreps::ParameterService::Service {
 public:
  ParameterServiceImpl(CachePS *cache_ps) { cache_ps_ = cache_ps; }

 private:
  Status GetParameter(ServerContext *context,
                      const GetParameterRequest *request,
                      GetParameterResponse *reply) override {
    base::ConstArray<uint64_t> keys_array(request->keys());
    bool isPerf = request->has_perf() && request->perf();
    if (isPerf) {
      xmh::PerfCounter::Record("PS Get Keys", keys_array.Size());
    }
    xmh::Timer timer_ps_get_req("PS GetParameter Req");
    ParameterCompressor compressor;
    std::vector<std::string> blocks;
    FB_LOG_EVERY_MS(INFO, 1000)
        << "[PS] Getting " << keys_array.Size() << " keys";

    xmh::Timer timer_cache_get("CachePS GetParameter");
    for (auto each : keys_array) {
      ParameterPack parameter_pack;
      cache_ps_->GetParameterRun2Completion(each, parameter_pack, 0);
      compressor.AddItem(parameter_pack, &blocks);
    }
    if (isPerf) {
      timer_cache_get.end();
    } else {
      timer_cache_get.destroy();
    }

    compressor.ToBlock(&blocks);
    CHECK_EQ(blocks.size(), 1);
    reply->mutable_parameter_value()->swap(blocks[0]);

    if (isPerf) {
      timer_ps_get_req.end();
    } else {
      timer_ps_get_req.destroy();
    }
    return Status::OK;
  }

  Status Command(ServerContext *context, const CommandRequest *request,
                 CommandResponse *reply) override {
    if (request->command() == PSCommand::CLEAR_PS) {
      LOG(WARNING) << "[PS Command] Clear All";
      cache_ps_->Clear();
    } else if (request->command() == PSCommand::RELOAD_PS) {
      LOG(WARNING) << "[PS Command] Reload PS";
      CHECK_NE(request->arg1().size(), 0);
      CHECK_NE(request->arg2().size(), 0);
      CHECK_EQ(request->arg1().size(), 1);
      LOG(WARNING) << "model_config_path = " << request->arg1()[0];
      for (int i = 0; i < request->arg2().size(); i++) {
        LOG(WARNING) << fmt::format("emb_file {}: {}", i, request->arg2()[i]);
      }
      std::vector<std::string> arg1;
      for (auto &each : request->arg1()) {
        arg1.push_back(each);
      }
      std::vector<std::string> arg2;
      for (auto &each : request->arg2()) {
        arg2.push_back(each);
      }

      cache_ps_->Initialize(arg1, arg2);
    } else {
      LOG(FATAL) << "invalid command";
    }
    return Status::OK;
  }

  Status PutParameter(ServerContext *context,
                      const PutParameterRequest *request,
                      PutParameterResponse *reply) override {
    const ParameterCompressReader *reader =
        reinterpret_cast<const ParameterCompressReader *>(
            request->parameter_value().data());
    int size = reader->item_size();
    for (int i = 0; i < size; i++) {
      cache_ps_->PutSingleParameter(reader->item(i), 0);
    }
    return Status::OK;
  }

 private:
  CachePS *cache_ps_;
};

namespace recstore {
class GRPCParameterServer : public BaseParameterServer {
 public:
  GRPCParameterServer() = default;

  void Run() {
    // 检查是否配置了多分片
    int num_shards = 1;  // 默认单分片
    if (config_["cache_ps"].contains("num_shards")) {
        num_shards = config_["cache_ps"]["num_shards"];
    }
    
    if (num_shards > 1) {
        // 多服务器启动逻辑
        std::cout << "启动分布式参数服务器，分片数量: " << num_shards << std::endl;
        
        if (!config_["cache_ps"].contains("servers")) {
            LOG(FATAL) << "配置了 num_shards > 1 但缺少 servers 配置";
            return;
        }
        
        auto servers = config_["cache_ps"]["servers"];
        if (servers.size() != num_shards) {
            LOG(FATAL) << "servers 配置数量 (" << servers.size() 
                      << ") 与 num_shards (" << num_shards << ") 不匹配";
            return;
        }
        
        std::vector<std::thread> server_threads;
        
        for (auto& server_config : servers) {
            server_threads.emplace_back([this, server_config]() {
                std::string host = server_config["host"];
                int port = server_config["port"];
                int shard = server_config["shard"];
                
                std::string server_address = host + ":" + std::to_string(port);
                auto cache_ps = std::make_unique<CachePS>(config_["cache_ps"]);
                ParameterServiceImpl service(cache_ps.get());
                
                grpc::EnableDefaultHealthCheckService(true);
                grpc::reflection::InitProtoReflectionServerBuilderPlugin();
                ServerBuilder builder;
                builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
                builder.RegisterService(&service);
                std::unique_ptr<Server> server(builder.BuildAndStart());
                std::cout << "Server shard " << shard << " listening on " << server_address << std::endl;
                server->Wait();
            });
        }
        
        // 等待所有服务器线程
        for (auto& t : server_threads) {
            t.join();
        }
    } else {
        // 单服务器启动逻辑
        std::cout << "启动单参数服务器" << std::endl;
        std::string server_address("0.0.0.0:15000");
        auto cache_ps = std::make_unique<CachePS>(config_["cache_ps"]);
        ParameterServiceImpl service(cache_ps.get());
        grpc::EnableDefaultHealthCheckService(true);
        grpc::reflection::InitProtoReflectionServerBuilderPlugin();
        ServerBuilder builder;
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
        builder.RegisterService(&service);
        std::unique_ptr<Server> server(builder.BuildAndStart());
        std::cout << "Server listening on " << server_address << std::endl;
        server->Wait();
    }
  }
};

FACTORY_REGISTER(BaseParameterServer, GRPCParameterServer, GRPCParameterServer);

}  // namespace recstore

int main(int argc, char **argv) {
  folly::Init(&argc, &argv);
  const char *prof = std::getenv("CPUPROFILE");
  if (prof && prof[0]) {
    ProfilerStart(prof);
  }
  xmh::Reporter::StartReportThread(2000);
  std::ifstream config_file(FLAGS_config_path);
  nlohmann::json ex;
  config_file >> ex;
  recstore::GRPCParameterServer ps;
  std::cout << "Parameter server config: " << ex.dump(2) << std::endl;
  ps.Init(ex);
  ps.Run();

  if (prof && prof[0]) {
    ProfilerStop();
  }
  return 0;
}