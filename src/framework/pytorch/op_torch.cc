#include <torch/extension.h>
#include "framework/op.h"
#include "base/tensor.h"
#include "base/timer.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <thread>
#include <atomic>
#include <fstream>
#include <chrono>
#include <iomanip>

namespace recstore {
namespace framework {

// Log level: 0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG
static int get_log_level() {
  static int level = []() {
    const char* env = std::getenv("RECSTORE_LOG_LEVEL");
    if (!env)
      return 2; // Default INFO
    return std::atoi(env);
  }();
  return level;
}
#define RECSTORE_LOG(level, msg)                                               \
  do {                                                                         \
    if (get_log_level() >= level) {                                            \
      std::cout << msg << std::endl;                                           \
    }                                                                          \
  } while (0)

static inline base::RecTensor
ToRecTensor(const torch::Tensor& tensor, base::DataType dtype) {
  std::vector<int64_t> shape;
  for (int i = 0; i < tensor.dim(); ++i) {
    shape.push_back(tensor.size(i));
  }
  return base::RecTensor(const_cast<void*>(tensor.data_ptr()), shape, dtype);
}

// ========== Optional trainer-side perf reporting (inside Python process) ==========
namespace {
std::once_flag g_perf_once;
std::atomic<bool> g_perf_stop{false};
std::unique_ptr<std::thread> g_perf_thread;
std::string g_perf_path;
int g_perf_interval_ms = 5000;

void AppendToFile(const std::string &path, const std::string &content) {
  std::ofstream ofs(path, std::ios::app);
  if (!ofs.is_open()) {
    RECSTORE_LOG(0, "[ERROR] [trainer-perf] open file failed: " << path);
    return;
  }
  ofs << content << std::endl;
}

void StartTrainerPerfReporterOnce() {
  std::call_once(g_perf_once, []() {
    const char* path_env = std::getenv("RECSTORE_PERF_REPORT_PATH");
    if (!path_env || !path_env[0]) {
      // Disabled unless user sets RECSTORE_PERF_REPORT_PATH
      return;
    }
    g_perf_path = path_env;
    if (const char* itv = std::getenv("RECSTORE_PERF_INTERVAL_MS")) {
      int v = std::atoi(itv);
      if (v > 0) g_perf_interval_ms = v;
    }
    g_perf_stop.store(false, std::memory_order_release);
    g_perf_thread = std::make_unique<std::thread>([]() {
      while (!g_perf_stop.load(std::memory_order_acquire)) {
        std::stringstream ss;
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        ss << "\n===== Trainer Perf Report @ "
           << std::put_time(std::localtime(&now), "%F %T") << " =====\n";
        ss << xmh::Timer::Report() << "\n" << xmh::PerfCounter::Report();
        AppendToFile(g_perf_path, ss.str());
        std::this_thread::sleep_for(std::chrono::milliseconds(g_perf_interval_ms));
      }
    });
    std::atexit([](){
      g_perf_stop.store(true, std::memory_order_release);
      if (g_perf_thread && g_perf_thread->joinable()) g_perf_thread->join();
      if (!g_perf_path.empty()) {
        std::stringstream ss;
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        ss << "\n===== Trainer Final Perf Report @ "
           << std::put_time(std::localtime(&now), "%F %T") << " =====\n";
        ss << xmh::Timer::Report() << "\n" << xmh::PerfCounter::Report();
        AppendToFile(g_perf_path, ss.str());
      }
    });
  });
}
} // anonymous namespace

torch::Tensor emb_read_torch(const torch::Tensor& keys, int64_t embedding_dim) {
  StartTrainerPerfReporterOnce();
  xmh::Timer t_total("OP.EmbRead.Total");
  RECSTORE_LOG(0,
               "[DEBUG][op_torch] emb_read_torch: keys shape="
                   << keys.sizes() << ", dtype=" << keys.dtype()
                   << ", data_ptr=" << keys.data_ptr());
  RECSTORE_LOG(
      0, "[DEBUG][op_torch] emb_read_torch: embedding_dim=" << embedding_dim);
  xmh::Timer t_prepare("OP.EmbRead.Prepare");
  torch::Tensor cpu_keys = keys;
  if (keys.is_cuda()) {
    xmh::Timer t_copy_k("OP.EmbRead.ToCPUKeys");
    RECSTORE_LOG(2, "[INFO] emb_read_torch: copying GPU keys to CPU");
    cpu_keys = keys.cpu();
    t_copy_k.end();
  }
  if (cpu_keys.size(0) > 0) {
    auto cpu_keys_acc = cpu_keys.accessor<int64_t, 1>();
    std::ostringstream oss;
    oss << "[DEBUG][op_torch] emb_read_torch: keys start with: ";
    for (int i = 0; i < std::min((int64_t)10, keys.size(0)); ++i)
      oss << cpu_keys_acc[i] << ", ";
    RECSTORE_LOG(0, oss.str());
  }
  RECSTORE_LOG(2,
               "[INFO] emb_read_torch called: keys shape="
                   << cpu_keys.sizes() << ", dtype=" << cpu_keys.dtype()
                   << ", embedding_dim=" << embedding_dim);
  TORCH_CHECK(cpu_keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(cpu_keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(cpu_keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");

  const int64_t num_keys = cpu_keys.size(0);
  if (num_keys == 0) {
    RECSTORE_LOG(3, "[DEBUG] emb_read_torch: num_keys==0, returning empty");
    return torch::empty(
        {0, embedding_dim}, cpu_keys.options().dtype(torch::kFloat32));
  }

  auto op = GetKVClientOp();

  auto values = torch::empty(
      {num_keys, embedding_dim}, keys.options().dtype(torch::kFloat32));
  torch::Tensor cpu_values = values;
  if (values.is_cuda()) {
    xmh::Timer t_copy_vbuf("OP.EmbRead.ToCPUValuesBuffer");
    RECSTORE_LOG(
        2,
        "[INFO] emb_read_torch: copying GPU values to CPU for C++ operation");
    cpu_values = values.cpu();
    t_copy_vbuf.end();
  }
  TORCH_CHECK(cpu_values.is_contiguous(),
              "Internal error: Created values tensor is not contiguous");
  RECSTORE_LOG(0,
               "[DEBUG][op_torch] emb_read_torch: values shape="
                   << cpu_values.sizes() << ", dtype=" << cpu_values.dtype()
                   << ", data_ptr=" << cpu_values.data_ptr());
  if (cpu_values.size(0) > 0) {
    auto values_acc = cpu_values.accessor<float, 2>();
    std::ostringstream oss;
    oss << "[DEBUG][op_torch] emb_read_torch: values start with: ";
    for (int i = 0; i < std::min((int64_t)10, cpu_values.size(0)); ++i) {
      oss << "[";
      for (int j = 0; j < std::min((int64_t)10, cpu_values.size(1)); ++j) {
        oss << values_acc[i][j] << ", ";
      }
      oss << "] ";
    }
    RECSTORE_LOG(0, oss.str());
  }

  base::RecTensor rec_keys   = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(cpu_values, base::DataType::FLOAT32);
  t_prepare.end();

  RECSTORE_LOG(3, "[DEBUG] emb_read_torch: calling op->EmbRead");
  {
    xmh::Timer t_call("OP.EmbRead.Call");
    op->EmbRead(rec_keys, rec_values);
    t_call.end();
  }
  RECSTORE_LOG(3, "[DEBUG] emb_read_torch: EmbRead done");

  if (values.is_cuda()) {
    xmh::Timer t_copy_back("OP.EmbRead.CopyBack");
    RECSTORE_LOG(2, "[INFO] emb_read_torch: copying results back to GPU");
    values.copy_(cpu_values);
    t_copy_back.end();
  }
  t_total.end();

  return values;
}

void emb_update_torch(const torch::Tensor& keys, const torch::Tensor& grads) {
  throw std::runtime_error(
      "emb_update_torch is deprecated. Use the Python-based sparse optimizer.");
}

void emb_write_torch(const torch::Tensor& keys, const torch::Tensor& values) {
  StartTrainerPerfReporterOnce();
  xmh::Timer t_total("OP.EmbWrite.Total");
  RECSTORE_LOG(0,
               "[DEBUG][op_torch] emb_write_torch: keys shape="
                   << keys.sizes() << ", dtype=" << keys.dtype()
                   << ", data_ptr=" << keys.data_ptr());
  RECSTORE_LOG(0,
               "[DEBUG][op_torch] emb_write_torch: values shape="
                   << values.sizes() << ", dtype=" << values.dtype()
                   << ", data_ptr=" << values.data_ptr());
  if (keys.size(0) > 0) {
    auto keys_acc = keys.accessor<int64_t, 1>();
    std::ostringstream oss;
    oss << "[DEBUG][op_torch] emb_write_torch: keys start with: ";
    for (int i = 0; i < std::min((int64_t)10, keys.size(0)); ++i)
      oss << keys_acc[i] << ", ";
    RECSTORE_LOG(0, oss.str());
  }
  if (values.size(0) > 0) {
    auto values_acc = values.accessor<float, 2>();
    std::ostringstream oss;
    oss << "[DEBUG][op_torch] emb_write_torch: values start with: ";
    for (int i = 0; i < std::min((int64_t)10, values.size(0)); ++i) {
      oss << "[";
      for (int j = 0; j < std::min((int64_t)10, values.size(1)); ++j) {
        oss << values_acc[i][j] << ", ";
      }
      oss << "] ";
    }
    RECSTORE_LOG(0, oss.str());
  }
  RECSTORE_LOG(2,
               "[INFO] emb_write_torch called: keys shape="
                   << keys.sizes() << ", values shape=" << values.sizes());
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(values.dim() == 2, "Values tensor must be 2-dimensional");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32,
              "Values tensor must have dtype float32");
  TORCH_CHECK(values.is_contiguous(), "Values tensor must be contiguous");
  TORCH_CHECK(keys.size(0) == values.size(0),
              "Keys and Values tensors must have the same number of entries");

  if (keys.size(0) == 0) {
    RECSTORE_LOG(3, "[DEBUG] emb_write_torch: num_keys==0, early return");
    return;
  }

  auto op = GetKVClientOp();

  xmh::Timer t_prepare("OP.EmbWrite.Prepare");
  torch::Tensor cpu_keys   = keys;
  torch::Tensor cpu_values = values;
  if (keys.is_cuda()) {
    xmh::Timer t_copy_k("OP.EmbWrite.ToCPUKeys");
    RECSTORE_LOG(2, "[INFO] emb_write_torch: copying GPU keys to CPU");
    cpu_keys = keys.cpu();
    t_copy_k.end();
  }
  if (values.is_cuda()) {
    xmh::Timer t_copy_v("OP.EmbWrite.ToCPUValues");
    RECSTORE_LOG(2, "[INFO] emb_write_torch: copying GPU values to CPU");
    cpu_values = values.cpu();
    t_copy_v.end();
  }

  base::RecTensor rec_keys   = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(cpu_values, base::DataType::FLOAT32);
  t_prepare.end();

  RECSTORE_LOG(3, "[DEBUG] emb_write_torch: calling op->EmbWrite");
  {
    xmh::Timer t_call("OP.EmbWrite.Call");
    op->EmbWrite(rec_keys, rec_values);
    t_call.end();
  }
  RECSTORE_LOG(3, "[DEBUG] emb_write_torch: EmbWrite done");
  t_total.end();
}

TORCH_LIBRARY(recstore_ops, m) {
  m.def("emb_read", emb_read_torch);
  m.def("emb_update", emb_update_torch);
  m.def("emb_write", emb_write_torch);
}

} // namespace framework
} // namespace recstore
