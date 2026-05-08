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
  endfunction()
  recstore_add_hps_rocksdb()
endif()

if(RECSTORE_HUGECTR_CUDA_HEADERS_AVAILABLE)
  add_executable(hps_backend_benchmark
    hps_backend_benchmark.cc
    hps_backend_shim.cc
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/core23/logger.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/thread_pool.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/hps/database_backend.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/hps/hash_map_backend.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/hps/rocksdb_backend.cpp
  )
  target_compile_definitions(hps_backend_benchmark PRIVATE HCTR_USE_ROCKS_DB)
  target_include_directories(hps_backend_benchmark PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/include
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/parallel-hashmap
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/rocksdb/include
    ${RECSTORE_CUDA_INCLUDE_DIRS}
  )
  target_link_libraries(hps_backend_benchmark PUBLIC
    pthread
    base
    kv_engine
    ps_common
    TBB::tbb
    rocksdb-shared
  )
  target_link_libraries(hps_backend_benchmark PUBLIC OpenMP::OpenMP_CXX)
else()
  message(STATUS "Skipping hps_backend_benchmark: HugeCTR CUDA headers are unavailable.")
endif()
