set(RECSTORE_HPS_ROCKSDB_DIR
  ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/rocksdb)

if(NOT TARGET rocksdb-shared)
  function(recstore_apply_hps_rocksdb_compat target_name)
    if(TARGET ${target_name})
      target_compile_options(${target_name} PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-include>"
        "$<$<COMPILE_LANGUAGE:CXX>:cstdint>"
        -UROCKSDB_MALLOC_USABLE_SIZE)
    endif()
  endfunction()

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
    recstore_apply_hps_rocksdb_compat(rocksdb)
    recstore_apply_hps_rocksdb_compat(rocksdb-shared)
  endfunction()
  recstore_add_hps_rocksdb()
endif()

if(RECSTORE_HUGECTR_CUDA_HEADERS_AVAILABLE)
  set(HUGECTR_GENERATED_INCLUDE_DIR
    ${PROJECT_BINARY_DIR}/third_party/HugeCTR/HugeCTR/include)
  file(MAKE_DIRECTORY ${HUGECTR_GENERATED_INCLUDE_DIR})
  configure_file(
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/include/config.hpp.in
    ${HUGECTR_GENERATED_INCLUDE_DIR}/config.hpp
    @ONLY
  )

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
    ${HUGECTR_GENERATED_INCLUDE_DIR}
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
