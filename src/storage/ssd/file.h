#pragma once

#include "spdk/env.h"
#include "spdk/nvme.h"
#include "spdk/vmd.h"
#include <boost/coroutine2/all.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <folly/Format.h>
#include <folly/GLog.h>
#include <folly/Likely.h>
#include <glog/logging.h>
#include <stdexcept>
#include <sys/user.h>

typedef uint64_t PageID_t;
const PageID_t INVALID_PAGE = -1;
using boost::coroutines2::coroutine;
extern thread_local int pending;
extern thread_local std::vector<std::unique_ptr<coroutine<void>::pull_type>>
    coros;
static const char* pcie_address = "0000:c2:00.0";

static void io_complete(void* arg, const struct spdk_nvme_cpl* cpl) {
  int64_t t = (int64_t)(intptr_t)arg;
  if (!spdk_nvme_cpl_is_success(cpl))
    fprintf(stderr, "I/O error!\n");
  pending--;
  if (t >= 0)
    (*coros[t])();
}

class FileManager {
public:
  FileManager(int queue_size) : next_page_id(0), queue_size(queue_size) {
    LOG(INFO) << "Initializing NVMe Controllers";
    spdk_env_opts_init(&opts);
    spdk_nvme_trid_populate_transport(&trid, SPDK_NVME_TRANSPORT_PCIE);
    snprintf(trid.subnqn, sizeof(trid.subnqn), "%s", SPDK_NVMF_DISCOVERY_NQN);
    snprintf(trid.traddr, sizeof(trid.traddr), "%s", pcie_address);
    CHECK(spdk_env_init(&opts) >= 0) << "Unable to initialize SPDK env\n";
    CHECK_EQ(spdk_vmd_init(), 0) << "Failed to initialize VMD";
    CHECK_EQ(spdk_nvme_probe(&trid, this, probe_cb, attach_cb, NULL), 0)
        << "Failed to probe NVMe device";
    CHECK_NE(ns_entry.ns, nullptr) << "Namespace is not initialized";
    CHECK_NE(ns_entry.ctrlr, nullptr) << "Controller is not initialized";
    empty_page = (char*)spdk_zmalloc(
        PAGE_SIZE, 64, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
  }

  ~FileManager() {
    struct spdk_nvme_detach_ctx* detach_ctx = NULL;
    spdk_nvme_detach_async(ns_entry.ctrlr, &detach_ctx);
    if (detach_ctx)
      spdk_nvme_detach_poll(detach_ctx);
  }

  PageID_t AllocatePage(coroutine<void>::push_type& sink, uint64_t index) {
    PageID_t new_page_id = next_page_id++;
    WritePageAsync(sink, index, new_page_id, empty_page);
    return new_page_id;
  }
  PageID_t AllocatePage() {
    PageID_t new_page_id = next_page_id++;
    WritePageSync(new_page_id, empty_page);
    return new_page_id;
  }

  void ReadPage(coroutine<void>::push_type& sink,
                uint64_t index,
                PageID_t page_id,
                char* buffer) {
    ReadPageAsync(sink, index, page_id, buffer);
  }
  void ReadPage(PageID_t page_id, char* buffer) {
    ReadPageSync(page_id, buffer);
  }

  void WritePage(coroutine<void>::push_type& sink,
                 uint64_t index,
                 PageID_t page_id,
                 char* buffer) {
    WritePageAsync(sink, index, page_id, buffer);
  }
  void WritePage(PageID_t page_id, char* buffer) {
    WritePageSync(page_id, buffer);
  }

  template <typename T>
  T* GetPage(coroutine<void>::push_type& sink,
             uint64_t index,
             PageID_t page_id) {
    char* buffer = (char*)spdk_zmalloc(
        PAGE_SIZE, 64, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
    CHECK_NE(buffer, nullptr) << "Failed to allocate memory for page";
    ReadPage(sink, index, page_id, buffer);
    return reinterpret_cast<T*>(buffer);
  }
  template <typename T>
  T* GetPage(PageID_t page_id) {
    char* buffer = (char*)spdk_zmalloc(
        PAGE_SIZE, 64, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
    CHECK_NE(buffer, nullptr) << "Failed to allocate memory for page";
    ReadPage(page_id, buffer);
    return reinterpret_cast<T*>(buffer);
  }

  // Unpin a page, if dirty, write it back
  void Unpin(coroutine<void>::push_type& sink,
             uint64_t index,
             PageID_t page_id,
             void* page_data,
             bool is_dirty) {
    if (is_dirty)
      WritePage(sink, index, page_id, reinterpret_cast<char*>(page_data));
    spdk_free(page_data);
  }
  void Unpin(PageID_t page_id, void* page_data, bool is_dirty) {
    if (is_dirty)
      WritePage(page_id, reinterpret_cast<char*>(page_data));
    spdk_free(page_data);
  }

  struct spdk_nvme_qpair* get_thread_qpair() {
    static thread_local ThreadQpair thread_qpair;
    return thread_qpair.get(ns_entry.ctrlr, queue_size);
  }

private:
  PageID_t next_page_id;
  char* empty_page = nullptr;
  int queue_size;

  spdk_env_opts opts;
  struct spdk_nvme_transport_id trid = {};
  struct ns_entry {
    struct spdk_nvme_ctrlr* ctrlr;
    struct spdk_nvme_ns* ns;
  } ns_entry;

  struct ThreadQpair {
    struct spdk_nvme_qpair* qpair = NULL;
    bool initialized              = false;
    ThreadQpair()                 = default;
    ~ThreadQpair() {
      if (initialized)
        spdk_nvme_ctrlr_free_io_qpair(qpair);
    }

    struct spdk_nvme_qpair* get(struct spdk_nvme_ctrlr* ctrlr, int queue_size) {
      if (!initialized) {
        CHECK_NE(ctrlr, nullptr) << "No NVMe controller";
        struct spdk_nvme_io_qpair_opts opts;
        spdk_nvme_ctrlr_get_default_io_qpair_opts(ctrlr, &opts, sizeof(opts));
        opts.io_queue_size     = queue_size + 1; // 你想要的队列深度
        opts.io_queue_requests = queue_size + 1; // 一般和队列深度一致
        qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctrlr, &opts, sizeof(opts));
        CHECK_NE(qpair, nullptr) << "Failed to allocate IO qpair";
        initialized = true;
      }
      return qpair;
    }
  };

  void ReadPageAsync(coroutine<void>::push_type& sink,
                     uint64_t index,
                     PageID_t page_id,
                     char* buffer) {
    struct spdk_nvme_qpair* qpair = get_thread_qpair();
    pending++;
    int ret = spdk_nvme_ns_cmd_read(
        ns_entry.ns,
        qpair,
        buffer,
        page_id * (PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns)),
        PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns),
        io_complete,
        (void*)index,
        0);
    if (ret == -ENOMEM)
      throw std::runtime_error("Failed to read page");
    sink();
  }
  void ReadPageSync(PageID_t page_id, char* buffer) {
    struct spdk_nvme_qpair* qpair = get_thread_qpair();
    pending++;
    int ret = spdk_nvme_ns_cmd_read(
        ns_entry.ns,
        qpair,
        buffer,
        page_id * (PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns)),
        PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns),
        io_complete,
        (void*)(-1),
        0);
    if (ret == -ENOMEM)
      throw std::runtime_error("Failed to read page");
    while (pending > 0) {
      spdk_nvme_qpair_process_completions(qpair, 0);
    }
  }

  void WritePageAsync(coroutine<void>::push_type& sink,
                      uint64_t index,
                      PageID_t page_id,
                      char* buffer) {
    struct spdk_nvme_qpair* qpair = get_thread_qpair();
    pending++;
    int ret = spdk_nvme_ns_cmd_write(
        ns_entry.ns,
        qpair,
        buffer,
        page_id * (PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns)),
        PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns),
        io_complete,
        (void*)index,
        0);
    if (ret == -ENOMEM)
      throw std::runtime_error("Failed to write page");
    sink();
  }
  void WritePageSync(PageID_t page_id, char* buffer) {
    struct spdk_nvme_qpair* qpair = get_thread_qpair();
    pending++;
    int ret = spdk_nvme_ns_cmd_write(
        ns_entry.ns,
        qpair,
        buffer,
        page_id * (PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns)),
        PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns),
        io_complete,
        (void*)(-1),
        0);
    if (ret == -ENOMEM)
      throw std::runtime_error("Failed to write page");
    while (pending > 0) {
      spdk_nvme_qpair_process_completions(qpair, 0);
    }
  }

  static bool probe_cb(void* cb_ctx,
                       const struct spdk_nvme_transport_id* trid,
                       struct spdk_nvme_ctrlr_opts* opts) {
    if (strcmp(pcie_address, trid->traddr) != 0)
      return false;
    LOG(INFO) << "Attaching to " << trid->traddr;
    return true;
  }

  static void attach_cb(void* cb_ctx,
                        const struct spdk_nvme_transport_id* trid,
                        struct spdk_nvme_ctrlr* ctrlr,
                        const struct spdk_nvme_ctrlr_opts* opts) {
    FileManager* ptr = (FileManager*)(cb_ctx);
    LOG(INFO) << fmt::format("Attached to {}", trid->traddr);
    for (uint32_t i = spdk_nvme_ctrlr_get_first_active_ns(ctrlr); i != 0;
         i          = spdk_nvme_ctrlr_get_next_active_ns(ctrlr, i)) {
      struct spdk_nvme_ns* ns = spdk_nvme_ctrlr_get_ns(ctrlr, i);
      if (ns && spdk_nvme_ns_is_active(ns)) {
        ptr->ns_entry.ctrlr = ctrlr;
        ptr->ns_entry.ns    = ns;
        LOG(INFO) << fmt::format(
            "Using NS {} size {}\n",
            i,
            (unsigned long)spdk_nvme_ns_get_size(ptr->ns_entry.ns));
        break;
      }
    }
  }
};