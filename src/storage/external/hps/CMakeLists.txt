set(RECSTORE_HPS_ROCKSDB_DIR
  ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/rocksdb)

if(NOT TARGET rocksdb-shared)
  function(recstore_add_hps_rocksdb)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(PORTABLE ON)
    set(WITH_TESTS OFF)
    set(WITH_BENCHMARK_TOOLS OFF)
    set(WITH_CORE_TOOLS OFF)
    set(WITH_TOOLS OFF)
    set(WITH_EXAMPLES OFF)
    set(WITH_FOLLY_DISTRIBUTED_MUTEX OFF)
    set(FAIL_ON_WARNINGS OFF)
    add_subdirectory(${RECSTORE_HPS_ROCKSDB_DIR}
      ${PROJECT_BINARY_DIR}/third_party/HugeCTR/rocksdb)
    if(TARGET rocksdb-shared)
      target_compile_options(rocksdb-shared PRIVATE -UROCKSDB_MALLOC_USABLE_SIZE)
    endif()
  endfunction()
  recstore_add_hps_rocksdb()
endif()

if(RECSTORE_HUGECTR_CUDA_HEADERS_AVAILABLE)
  add_library(hps_native_support STATIC
    hps_shim.cc
    raw_rocksdb.cc
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/core23/logger.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/thread_pool.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/hps/database_backend.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/hps/hash_map_backend.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/hps/rocksdb_backend.cpp
  )
  target_compile_definitions(hps_native_support PUBLIC HCTR_USE_ROCKS_DB)
  target_include_directories(hps_native_support PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/include
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/parallel-hashmap
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/rocksdb/include
    ${RECSTORE_CUDA_INCLUDE_DIRS}
  )
  target_link_libraries(hps_native_support PUBLIC
    pthread
    base
    TBB::tbb
    rocksdb-shared
    OpenMP::OpenMP_CXX
  )
  add_library(hps_backend_support INTERFACE)
  target_link_libraries(hps_backend_support INTERFACE
    hps_native_support
    kv_engine
    ps_common
  )
else()
  message(STATUS "Skipping hps_backend_support: HugeCTR CUDA headers are unavailable.")
endif()
