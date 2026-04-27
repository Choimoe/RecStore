#pragma once

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef ENABLE_PERF_REPORT
#  include "base/report/report_client.h"
#endif

namespace recstore {

struct LocalShmStageReportContext {
  bool active             = false;
  uint64_t request_id     = 0;
  uint32_t opcode         = 0;
  uint32_t ready_queue_id = 0;
  uint32_t key_count      = 0;
  uint32_t embedding_dim  = 0;
};

inline thread_local LocalShmStageReportContext g_local_shm_stage_report_context;

inline bool IsLocalShmStageReportEnabled() {
#ifdef ENABLE_PERF_REPORT
  static const bool enabled = []() {
    const char* env = std::getenv("RECSTORE_LOCAL_SHM_STAGE_REPORT");
    return env != nullptr && std::strcmp(env, "0") != 0 &&
           std::strcmp(env, "false") != 0 && std::strcmp(env, "FALSE") != 0;
  }();
  return enabled;
#else
  return false;
#endif
}

inline std::string LocalShmStageReportUniqueId() {
  return "local_shm_req|" +
         std::to_string(g_local_shm_stage_report_context.request_id);
}

inline void ReportLocalShmStageMetric(const char* metric_name, double value) {
#ifdef ENABLE_PERF_REPORT
  if (!IsLocalShmStageReportEnabled() ||
      !g_local_shm_stage_report_context.active) {
    return;
  }
  const std::string unique_id = LocalShmStageReportUniqueId();
  report("local_shm_server_stages", unique_id.c_str(), metric_name, value);
#else
  (void)metric_name;
  (void)value;
#endif
}

inline void ReportLocalShmStageMetadataOnce() {
  if (!IsLocalShmStageReportEnabled() ||
      !g_local_shm_stage_report_context.active) {
    return;
  }
  ReportLocalShmStageMetric(
      "opcode", static_cast<double>(g_local_shm_stage_report_context.opcode));
  ReportLocalShmStageMetric(
      "ready_queue_id",
      static_cast<double>(g_local_shm_stage_report_context.ready_queue_id));
  ReportLocalShmStageMetric(
      "key_count",
      static_cast<double>(g_local_shm_stage_report_context.key_count));
  ReportLocalShmStageMetric(
      "embedding_dim",
      static_cast<double>(g_local_shm_stage_report_context.embedding_dim));
}

class LocalShmStageReportScope {
public:
  LocalShmStageReportScope(
      uint64_t request_id,
      uint32_t opcode,
      uint32_t ready_queue_id,
      uint32_t key_count,
      uint32_t embedding_dim)
      : previous_(g_local_shm_stage_report_context) {
    if (!IsLocalShmStageReportEnabled()) {
      return;
    }
    g_local_shm_stage_report_context.active         = true;
    g_local_shm_stage_report_context.request_id     = request_id;
    g_local_shm_stage_report_context.opcode         = opcode;
    g_local_shm_stage_report_context.ready_queue_id = ready_queue_id;
    g_local_shm_stage_report_context.key_count      = key_count;
    g_local_shm_stage_report_context.embedding_dim  = embedding_dim;
    ReportLocalShmStageMetadataOnce();
  }

  ~LocalShmStageReportScope() {
    if (!IsLocalShmStageReportEnabled()) {
      return;
    }
    g_local_shm_stage_report_context = previous_;
  }

private:
  LocalShmStageReportContext previous_{};
};

inline double
LocalShmElapsedUs(const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
             std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace recstore
