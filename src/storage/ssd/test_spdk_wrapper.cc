#include <glog/logging.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include "spdk_wrapper.h"

TEST(SpdkWrapper, test) {
  auto ssd = ssdps::SpdkWrapper::create(1);
  ssd->Init();

  char* buf = (char*)spdk_zmalloc(
      0x1000, 0x1000, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
  char* buf_2 = (char*)spdk_zmalloc(
      0x1000, 0x1000, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
  snprintf(buf, 0x1000, "%s", "Hello world!\n");

  ssd->SyncWrite((void*)buf, 0x1000, 0, 0);
  ssd->SyncRead((void*)buf_2, 0x1000, 0, 0);

  CHECK(strcmp(buf, buf_2) == 0);
}

// TEST(SpdkWrapper, test_fused_operation) {
//   auto ssd = ssdps::SpdkWrapper::create();
//   ssd->Init();
//   char *buf = (char *)spdk_zmalloc(0x1000, 0x1000, NULL,
//   SPDK_ENV_SOCKET_ID_ANY,
//                                    SPDK_MALLOC_DMA);
//   char *buf_2 = (char *)spdk_zmalloc(0x1000, 0x1000, NULL,
//                                      SPDK_ENV_SOCKET_ID_ANY,
//                                      SPDK_MALLOC_DMA);
//   snprintf(buf, 0x1000, "%s", "Hello world!\n");

//   ssd->SyncWrite((void *)buf, 512, 0);
//   ssd->SyncWrite((void *)buf, 512, 1);
//   // ssd->Sync2Read((void *)buf_2, 0);
//   std::this_thread::sleep_for(std::chrono::seconds(2));
//   CHECK(memcmp(buf, buf_2, 512) == 0);
//   CHECK(memcmp(buf, buf_2 + 512, 512) == 0);
// }

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
