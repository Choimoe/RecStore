#include <unistd.h>

#include <atomic>
#include <ctime>
#include <memory>
#include <thread>

#include "base/bind_core.h"
#include "base/factory.h"
#include "base/init.h"
#include "base/log.h"
#include "base/random.h"
#include "base/string.h"
#include "base/timer.h"
#include "base/zipf.h"
#include "benchmark/sample_reader.h"
#include "load_db.h"
#include "memory/shm_file.h"
#include "storage/kv_engine/base_kv.h"

DEFINE_string(db, "KVEngineExtendibleHash", "");
DEFINE_int64(key_space_m, 10, "key space in million");
DEFINE_string(dataset, "zipfian", "zipfian/dataset");

DEFINE_double(warmup_ratio,
              0.8,
              "bulk load (warmup_ratio * key_space) kvs in DB");
DEFINE_int32(warmup_thread_num, 36, "");
DEFINE_bool(preload, true, "whether to preload the DB");

DEFINE_int32(thread_num, 1, "");
DEFINE_int32(numa_id, 0, "");
DEFINE_bool(use_dram, true, "");

DEFINE_int32(value_size, 32 * 4, "");
DEFINE_double(zipf_theta, 0.99, "");

DEFINE_int32(running_seconds, 100, "");
DEFINE_int32(read_ratio, 100, "");

DEFINE_int32(batch_read_count, 300, "");

const int kMaxThread = 32;
std::thread th[kMaxThread];
uint64_t tp[kMaxThread][8];

std::atomic_bool start_flag;

BaseKV* base_kv;

std::atomic<bool> stop_flag;

void thread_run(int tid, SampleReader* sample) {
  base::auto_bind_core();

  base::PseudoRandom random_engine(tid);

  tp[tid][0] = 0;
  while (!start_flag)
    ;

  std::vector<uint64_t> keys_buffer(FLAGS_batch_read_count);
  xmh::Timer read_timer("batch get", 1000);
  while (!stop_flag) {
    read_timer.start();
    auto keys = sample->fillArray(&keys_buffer[0]);
    if (base::Random::rand32(100) < FLAGS_read_ratio) {
      std::vector<base::ConstArray<float>> values;
      base_kv->BatchGet(keys.ToConstArray(), &values, tid);
      read_timer.end();
    } else {
      for (int i = 0; i < keys.Size(); i++) {
        base_kv->Put(keys[i], random_engine.GetString(FLAGS_value_size), tid);
      }
    }
    tp[tid][0]++;
  }
}

int main(int argc, char* argv[]) {
  base::Init(&argc, &argv);
  xmh::Reporter::StartReportThread();

  std::string path;

  if (FLAGS_use_dram)
    path = base::SFormat("/dev/shm/");
  else
    path = base::SFormat("/media/aep{}/", FLAGS_numa_id);

  bool shuffle_load                                = false;
  base::PMMmapRegisterCenter::GetConfig().use_dram = FLAGS_use_dram;
  base::PMMmapRegisterCenter::GetConfig().numa_id  = FLAGS_numa_id;

  base::global_socket_id = FLAGS_numa_id;
  LOG(INFO) << "set NUMA ID = " << FLAGS_numa_id;

  BaseKVConfig config;
  config.json_config_["path"]       = path + FLAGS_db;
  config.json_config_["capacity"]   = FLAGS_key_space_m * 1024 * 1024LL;
  config.json_config_["value_size"] = FLAGS_value_size;
  config.num_threads_ = std::max(FLAGS_thread_num, FLAGS_warmup_thread_num);
  base::RegisterKVEngineFactories();
  base_kv =
      base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(FLAGS_db, config);

  LoadDBHelper load_db_helper(
      base_kv,
      0,
      FLAGS_warmup_thread_num,
      FLAGS_key_space_m * 1024 * 1024LL * FLAGS_warmup_ratio,
      FLAGS_value_size);
  if (FLAGS_preload) {
    load_db_helper.PreLoadDB(shuffle_load);
    base_kv->Util();
    load_db_helper.CheckDBLoad();
  }

  // PetDatasetReader *dataset_reader = nullptr;
  // if (FLAGS_dataset == "dataset") {
  //   dataset_reader = new PetDatasetReader(
  //       FLAGS_thread_num, FLAGS_key_space_m * FLAGS_warmup_ratio,
  //       "/data/project/kuai/dump.2022.08.17/sign*", true);
  // }

  std::vector<std::unique_ptr<SampleReader>> sample_readers;
  for (int _ = 0; _ < FLAGS_thread_num; _++) {
    SampleReader* each;
    if (FLAGS_dataset == "zipfian")
      each = new ZipfianSampleReader(
          _,
          FLAGS_key_space_m * FLAGS_warmup_ratio,
          FLAGS_zipf_theta,
          FLAGS_batch_read_count);
    else if (FLAGS_dataset == "dataset") {
      LOG(FATAL) << "We can not make the production dataset public.";
      // each = new PetDatasetSampleReader(dataset_reader, _,
      //                                   FLAGS_key_space_m *
      //                                   FLAGS_warmup_ratio,
      //                                   FLAGS_batch_read_count);
    } else {
      LOG(FATAL) << "dataset_ = " << FLAGS_dataset;
    }
    sample_readers.emplace_back(each);
  }

  for (int i = 0; i < FLAGS_thread_num; i++) {
    tp[i][0] = -1;
  }
  for (int i = 0; i < FLAGS_thread_num; i++) {
    th[i] = std::thread(thread_run, i, sample_readers[i].get()); // read_thread
  }

  CHECK_LE(FLAGS_thread_num, kMaxThread);

  timespec s, e;
  uint64_t pre_tp = 0;

  for (int i = 0; i < FLAGS_thread_num; i++) {
    while (tp[i][0] != 0)
      ;
  }

  start_flag = true;

  int running_seconds = 0;
  while (true) {
    clock_gettime(CLOCK_REALTIME, &s);
    sleep(1);
    clock_gettime(CLOCK_REALTIME, &e);
    int microseconds = (e.tv_sec - s.tv_sec) * 1000000 +
                       (double)(e.tv_nsec - s.tv_nsec) / 1000;

    uint64_t all_tp = 0;
    for (int i = 0; i < FLAGS_thread_num; ++i) {
      all_tp += tp[i][0];
    }
    uint64_t cap = all_tp - pre_tp;
    pre_tp       = all_tp;

    printf("throughput %.4f Mreq/s %.4f Mkv/s\n",
           cap * 1.0 / microseconds,
           cap * (double)FLAGS_batch_read_count / microseconds);
    running_seconds++;

    if (running_seconds == FLAGS_running_seconds) {
      stop_flag = true;
      for (int i = 0; i < FLAGS_thread_num; i++)
        th[i].join();
      break;
    }
  }
  LOG(INFO) << "gracefully exit";

  return 0;
}
