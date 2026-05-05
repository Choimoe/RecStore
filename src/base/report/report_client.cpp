#include "report_client.h"
#include "base/config.h"
#include "base/timer.h"
#include <glog/logging.h>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <filesystem>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstring>
#include <system_error>
#include <fstream>

#include <chrono>

using json = nlohmann::json;

namespace recstore {
thread_local uint64_t g_trace_id = 0;
}

static std::string GetApiUrl() {
  std::string default_url = "http://127.0.0.1:8081/report";
  try {
    const auto config_file = base::FindRecStoreConfigPath();
    if (!config_file.has_value()) {
      return default_url;
    }
    std::ifstream ifs(*config_file);
    if (ifs.is_open()) {
      nlohmann::json j;
      ifs >> j;
      if (j.contains("report_API")) {
        return j["report_API"];
      }
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "Error reading recstore_config.json: " << e.what();
  }
  return default_url;
}

namespace {

enum class ReportMode {
  kRemote,
  kLocal,
};

enum class LocalSinkMode {
  kGlog,
  kJsonl,
  kBoth,
};

struct LocalReportConfig {
  LocalSinkMode sink_mode;
  std::string jsonl_path;
  size_t flush_every_n;
};

struct StructuredReportEvent {
  const char* table_name;
  const char* unique_id;
  const char* metric_name;
  double metric_value;
  uint64_t timestamp_us;
  const char* source;
};

const LocalReportConfig& ResolveLocalReportConfig();

std::string ToLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return std::tolower(c);
      });
  return value;
}

bool ParseLocalReportMode(const std::string& value) {
  const std::string normalized = ToLower(value);
  return normalized == "local" || normalized == "off" ||
         normalized == "disable" || normalized == "disabled" ||
         normalized == "false" || normalized == "0";
}

bool ParseRemoteReportMode(const std::string& value) {
  const std::string normalized = ToLower(value);
  return normalized == "grafana" || normalized == "remote" ||
         normalized == "on" || normalized == "true" || normalized == "1";
}

bool IsLocalJsonlEnabled() {
  const auto mode = ResolveLocalReportConfig().sink_mode;
  return mode == LocalSinkMode::kJsonl || mode == LocalSinkMode::kBoth;
}

LocalSinkMode ResolveLocalSinkModeFromEnvValue(const char* value) {
  if (value[0] != '\0') {
    const std::string mode = ToLower(value);
    if (mode == "jsonl") {
      return LocalSinkMode::kJsonl;
    }
    if (mode == "both") {
      return LocalSinkMode::kBoth;
    }
  }
  return LocalSinkMode::kGlog;
}

size_t ResolveLocalFlushEveryNFromEnvValue(const char* value) {
  if (value[0] != '\0') {
    char* end                  = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end != value && *end == '\0' && parsed > 0) {
      return static_cast<size_t>(parsed);
    }
    LOG(WARNING) << "Invalid RECSTORE_REPORT_FLUSH_EVERY_N=" << value
                 << ", fallback to 1.";
  }
  return 1;
}

const LocalReportConfig& ResolveLocalReportConfig() {
  thread_local bool initialized = false;
  thread_local std::string cached_sink_env;
  thread_local std::string cached_path_env;
  thread_local std::string cached_flush_env;
  thread_local LocalReportConfig cached_config{
      LocalSinkMode::kGlog, "recstore_report_events.jsonl", 1};

  const char* sink_env    = std::getenv("RECSTORE_REPORT_LOCAL_SINK");
  const char* path_env    = std::getenv("RECSTORE_REPORT_JSONL_PATH");
  const char* flush_env   = std::getenv("RECSTORE_REPORT_FLUSH_EVERY_N");
  const char* sink_value  = sink_env == nullptr ? "" : sink_env;
  const char* path_value  = path_env == nullptr ? "" : path_env;
  const char* flush_value = flush_env == nullptr ? "" : flush_env;

  if (!initialized || cached_sink_env != sink_value ||
      cached_path_env != path_value || cached_flush_env != flush_value) {
    cached_config.sink_mode = ResolveLocalSinkModeFromEnvValue(sink_value);
    cached_config.jsonl_path =
        path_value[0] == '\0' ? "recstore_report_events.jsonl" : path_value;
    cached_config.flush_every_n =
        ResolveLocalFlushEveryNFromEnvValue(flush_value);
    cached_sink_env  = sink_value;
    cached_path_env  = path_value;
    cached_flush_env = flush_value;
    initialized      = true;
  }
  return cached_config;
}

uint64_t GetTimestampUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

void AppendJsonString(std::string& output, const char* value) {
  output.push_back('"');
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(value);
       *p != '\0';
       ++p) {
    switch (*p) {
    case '"':
      output.append("\\\"");
      break;
    case '\\':
      output.append("\\\\");
      break;
    case '\b':
      output.append("\\b");
      break;
    case '\f':
      output.append("\\f");
      break;
    case '\n':
      output.append("\\n");
      break;
    case '\r':
      output.append("\\r");
      break;
    case '\t':
      output.append("\\t");
      break;
    default:
      if (*p < 0x20) {
        static constexpr char kHex[] = "0123456789abcdef";
        output.append("\\u00");
        output.push_back(kHex[*p >> 4]);
        output.push_back(kHex[*p & 0x0F]);
      } else {
        output.push_back(static_cast<char>(*p));
      }
      break;
    }
  }
  output.push_back('"');
}

template <typename T>
void AppendNumber(std::string& output, T value) {
  char buffer[64];
  auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
  if (result.ec == std::errc()) {
    output.append(buffer, result.ptr);
  } else {
    output.append(std::to_string(value));
  }
}

std::string ToJsonLine(const StructuredReportEvent& event) {
  std::string output;
  output.reserve(
      160 + std::strlen(event.table_name) + std::strlen(event.unique_id) +
      std::strlen(event.metric_name) + std::strlen(event.source));
  output.append("{\"table_name\":");
  AppendJsonString(output, event.table_name);
  output.append(",\"unique_id\":");
  AppendJsonString(output, event.unique_id);
  output.append(",\"metric_name\":");
  AppendJsonString(output, event.metric_name);
  output.append(",\"metric_value\":");
  AppendNumber(output, event.metric_value);
  output.append(",\"timestamp_us\":");
  AppendNumber(output, event.timestamp_us);
  output.append(",\"source\":");
  AppendJsonString(output, event.source);
  output.push_back('}');
  return output;
}

bool IsJsonlSinkMode(LocalSinkMode mode) {
  return mode == LocalSinkMode::kJsonl || mode == LocalSinkMode::kBoth;
}

bool IsGlogSinkMode(LocalSinkMode mode) {
  return mode == LocalSinkMode::kGlog || mode == LocalSinkMode::kBoth;
}

class LocalJsonlSink {
public:
  static LocalJsonlSink& GetInstance() {
    static LocalJsonlSink instance;
    return instance;
  }

  void WriteLine(const LocalReportConfig& config, const std::string& line) {
    std::lock_guard<std::mutex> lock(mtx_);
    ConfigureIfNeeded(config);
    if (!ofs_.is_open()) {
      return;
    }
    ofs_ << line << '\n';
    ++writes_since_flush_;
    if (writes_since_flush_ >= config.flush_every_n) {
      ofs_.flush();
      writes_since_flush_ = 0;
    }
  }

private:
  LocalJsonlSink() = default;

  ~LocalJsonlSink() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (ofs_.is_open()) {
      ofs_.flush();
      ofs_.close();
    }
  }

  void ConfigureIfNeeded(const LocalReportConfig& config) {
    if (configured_ && jsonl_path_ == config.jsonl_path &&
        flush_every_n_ == config.flush_every_n) {
      return;
    }

    if (ofs_.is_open()) {
      ofs_.flush();
      ofs_.close();
    }
    writes_since_flush_ = 0;

    const std::filesystem::path output_path(config.jsonl_path);
    const auto parent = output_path.parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
    ofs_.open(output_path, std::ios::app);
    if (!ofs_.is_open()) {
      LOG(ERROR) << "Failed to open local report JSONL sink: "
                 << config.jsonl_path;
    }

    configured_    = true;
    jsonl_path_    = config.jsonl_path;
    flush_every_n_ = config.flush_every_n;
  }

  std::mutex mtx_;
  std::ofstream ofs_;
  bool configured_ = false;
  std::string jsonl_path_;
  size_t flush_every_n_      = 1;
  size_t writes_since_flush_ = 0;
};

void WriteLocalStructuredEvent(const std::string& serialized) {
  const auto config = ResolveLocalReportConfig();

  if (IsGlogSinkMode(config.sink_mode)) {
    LOG(INFO) << "REPORT_LOCAL_EVENT " << serialized;
  }

  if (IsJsonlSinkMode(config.sink_mode)) {
    LocalJsonlSink::GetInstance().WriteLine(config, serialized);
  }
}

bool TryRecordLatencyMetricToTimer(const StructuredReportEvent& event) {
  double value_ns = 0.0;
  if (std::strcmp(event.metric_name, "duration_ns") == 0 ||
      std::strcmp(event.metric_name, "latency_ns") == 0) {
    value_ns = event.metric_value;
  } else if (std::strcmp(event.metric_name, "duration_us") == 0 ||
             std::strcmp(event.metric_name, "latency_us") == 0) {
    value_ns = event.metric_value * 1000.0;
  } else {
    return false;
  }

  thread_local std::string cached_table;
  thread_local std::string cached_metric;
  thread_local std::string cached_timer_name;
  if (cached_table != event.table_name || cached_metric != event.metric_name) {
    cached_table      = event.table_name;
    cached_metric     = event.metric_name;
    cached_timer_name = cached_table + "." + cached_metric;
  }
  xmh::Timer::ManualRecordNs(cached_timer_name, value_ns);
  return true;
}

ReportMode ResolveReportMode() {
  if (const char* env_mode = std::getenv("RECSTORE_REPORT_MODE");
      env_mode != nullptr) {
    thread_local bool initialized = false;
    thread_local std::string cached_env_mode;
    thread_local ReportMode cached_mode = ReportMode::kRemote;
    if (!initialized || cached_env_mode != env_mode) {
      const std::string mode = env_mode;
      if (ParseLocalReportMode(mode)) {
        cached_env_mode = env_mode;
        cached_mode     = ReportMode::kLocal;
      } else if (ParseRemoteReportMode(mode)) {
        cached_env_mode = env_mode;
        cached_mode     = ReportMode::kRemote;
      } else {
        LOG(WARNING) << "Unknown RECSTORE_REPORT_MODE=" << mode
                     << ", fallback to report_API based behavior.";
        const std::string api_url = GetApiUrl();
        return api_url.empty() ? ReportMode::kLocal : ReportMode::kRemote;
      }
      initialized = true;
    }
    return cached_mode;
  }

  const std::string api_url = GetApiUrl();
  if (api_url.empty()) {
    return ReportMode::kLocal;
  }

  return ReportMode::kRemote;
}

bool IsRemoteReportEnabled() {
  return ResolveReportMode() == ReportMode::kRemote;
}

} // namespace

size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
  ((std::string*)userp)->append((char*)contents, size * nmemb);
  return size * nmemb;
}

class AsyncReportQueue {
public:
  static AsyncReportQueue& GetInstance() {
    static AsyncReportQueue instance;
    return instance;
  }

  void Enqueue(const std::string& payload) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      if (queue_.size() >= 10000) {
        queue_.pop();
      }
      queue_.push(payload);
    }
    cv_.notify_one();
  }

private:
  AsyncReportQueue() : stop_(false) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    worker_ = std::thread(&AsyncReportQueue::WorkerLoop, this);
  }

  ~AsyncReportQueue() {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    curl_global_cleanup();
  }

  void WorkerLoop() {
    CURL* curl = curl_easy_init();
    if (!curl)
      return;

    std::string api_url = GetApiUrl();

    struct curl_slist* headers =
        curl_slist_append(nullptr, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_URL, api_url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3L);

    std::string response_buffer;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_buffer);

    while (true) {
      std::string payload;
      {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]() { return stop_ || !queue_.empty(); });

        if (stop_ && queue_.empty()) {
          break;
        }

        payload = std::move(queue_.front());
        queue_.pop();
      }

      response_buffer.clear();
      curl_easy_setopt(curl, CURLOPT_POST, 1L);
      curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
      curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, payload.length());

      CURLcode res = curl_easy_perform(curl);
      if (res != CURLE_OK) {
        LOG(ERROR) << "CURL perform failed: " << curl_easy_strerror(res);
      } else {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code >= 200 && http_code < 300) {
          LOG(INFO) << "Report HTTP API success (HTTP Code: " << http_code
                    << ").";
        } else {
          LOG(ERROR) << "API failed (HTTP Code: " << http_code << ").";
        }
        DLOG(INFO) << "Server response: " << response_buffer;
      }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
  }

  std::queue<std::string> queue_;
  std::mutex mtx_;
  std::condition_variable cv_;
  bool stop_;
  std::thread worker_;
};

bool send_json_request(const std::string& json_payload) {
  if (!IsRemoteReportEnabled()) {
    DLOG(INFO) << "REPORT_LIB INFO: Remote reporting disabled; drop payload.";
    return true;
  }
  AsyncReportQueue::GetInstance().Enqueue(json_payload);
  return true;
}

bool is_report_remote_enabled_for_test() { return IsRemoteReportEnabled(); }

bool is_report_local_jsonl_enabled_for_test() { return IsLocalJsonlEnabled(); }

extern "C" bool
report(const char* table_name,
       const char* unique_id,
       const char* metric_name,
       double metric_value) {
  const StructuredReportEvent event = {
      .table_name   = table_name,
      .unique_id    = unique_id,
      .metric_name  = metric_name,
      .metric_value = metric_value,
      .timestamp_us = GetTimestampUs(),
      .source       = "report"};

  std::string json_payload = ToJsonLine(event);

  WriteLocalStructuredEvent(json_payload);
  TryRecordLatencyMetricToTimer(event);

  bool success = send_json_request(json_payload);
  if (success) {
    DLOG(INFO) << "REPORT_LIB INFO: Data for ID [" << unique_id
               << "] enqueued successfully.";
  }

  return success;
}

extern "C" bool report_flame_graph(
    const char* table_name, const char* unique_id, const FlameGraphData& data) {
  std::string combined_uid = std::string(unique_id) + "|" + data.label;

  auto now        = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);
  struct tm gmt_tm;
  gmtime_r(&now_time_t, &gmt_tm);
  gmt_tm.tm_hour           = 0;
  gmt_tm.tm_min            = 0;
  gmt_tm.tm_sec            = 0;
  auto start_of_day_s      = timegm(&gmt_tm);
  uint64_t start_of_day_us = static_cast<uint64_t>(start_of_day_s) * 1000000;

  double adjusted_start = data.start - static_cast<double>(start_of_day_us);

  report(table_name, combined_uid.c_str(), "level", data.level);
  report(table_name, combined_uid.c_str(), "value", data.value);
  report(table_name, combined_uid.c_str(), "self", data.self);
  report(table_name, combined_uid.c_str(), "start", adjusted_start);

  DLOG(INFO) << "REPORT_LIB INFO: Flame Graph Data for ID [" << unique_id
             << ", Label " << data.label << "] enqueued successfully as '"
             << combined_uid << "'. Adjusted start: " << adjusted_start;

  return true;
}

ReportTimeline::ReportTimeline(const std::string& table_name,
                               const std::string& unique_id,
                               const std::string& metric_name)
    : table_name_(table_name),
      unique_id_(unique_id),
      metric_name_(metric_name),
      is_done_(false) {
  report(table_name_.c_str(),
         unique_id_.c_str(),
         metric_name_.c_str(),
         static_cast<double>(STATE_RUNNING));
}

ReportTimeline::~ReportTimeline() {
  if (!is_done_) {
    done(STATE_SUCCESS);
  }
}

void ReportTimeline::done(TimelineState state) {
  if (!is_done_) {
    report(table_name_.c_str(),
           unique_id_.c_str(),
           metric_name_.c_str(),
           static_cast<double>(state));
    is_done_ = true;
  }
}
