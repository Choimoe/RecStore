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
  if(NOT TARGET rdkafka)
    cmake_policy(PUSH)
    cmake_policy(SET CMP0077 NEW)
    set(RDKAFKA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(RDKAFKA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(WITH_CURL OFF CACHE BOOL "" FORCE)
    add_subdirectory(
      ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/librdkafka
      ${PROJECT_BINARY_DIR}/third_party/HugeCTR/librdkafka)
    cmake_policy(POP)
  endif()

  if(NOT TARGET redis++)
    function(recstore_add_hps_redis)
      cmake_policy(PUSH)
      cmake_policy(SET CMP0077 NEW)
      set(ENABLE_SSL ON CACHE BOOL "" FORCE)
      set(DISABLE_TESTS ON CACHE BOOL "" FORCE)
      add_subdirectory(
        ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/hiredis
        ${PROJECT_BINARY_DIR}/third_party/HugeCTR/hiredis)

      set(stub_dir "${PROJECT_BINARY_DIR}/third_party/HugeCTR/generated/pkg")
      file(WRITE "${stub_dir}/hiredis-config.cmake" "")
      file(WRITE "${stub_dir}/hiredis_ssl-config.cmake" "")

      set(hiredis_DIR ${stub_dir})
      set(hiredis_LIBRARIES hiredis::hiredis)
      set(hiredis_INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/hiredis")
      if(NOT TARGET hiredis::hiredis)
        add_library(hiredis::hiredis ALIAS hiredis)
      endif()

      set(hiredis_ssl_DIR ${stub_dir})
      set(hiredis_ssl_LIBRARIES hiredis::hiredis_ssl)
      set(hiredis_ssl_INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/hiredis")
      if(NOT TARGET hiredis::hiredis_ssl)
        add_library(hiredis::hiredis_ssl ALIAS hiredis_ssl)
      endif()

      include_directories(${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party)
      set(REDIS_PLUS_PLUS_BUILD_STATIC OFF CACHE BOOL "" FORCE)
      set(REDIS_PLUS_PLUS_BUILD_TEST OFF CACHE BOOL "" FORCE)
      set(REDIS_PLUS_PLUS_USE_TLS ON CACHE BOOL "" FORCE)
      add_subdirectory(
        ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/redis_pp
        ${PROJECT_BINARY_DIR}/third_party/HugeCTR/redis_pp)
      cmake_policy(POP)
    endfunction()
    recstore_add_hps_redis()
  endif()

  add_library(hps_native_support STATIC
    hps_embedding_cache_noop_shim.cc
    hps_inference_params_shim.cc
    hps_raw_model_loader_shim.cc
    hps_shim.cc
    raw_rocksdb.cc
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/core23/logger.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/thread_pool.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/io/filesystem.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/io/local_filesystem.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/hps/database_backend.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/hps/hash_map_backend.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/hps/hier_parameter_server.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/hps/kafka_message.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/hps/message.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/hps/mp_hash_map_backend.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/hps/redis_backend.cpp
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/hps/rocksdb_backend.cpp
  )
  target_compile_definitions(hps_native_support PUBLIC HCTR_USE_ROCKS_DB)
  target_include_directories(hps_native_support PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/include
    ${PROJECT_BINARY_DIR}/third_party/HugeCTR/HugeCTR/include
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/embedding_cache/include
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/gpu_cache/include
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/librdkafka/src
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/parallel-hashmap
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/rocksdb/include
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/redis_pp/src
    ${RECSTORE_CUDA_INCLUDE_DIRS}
  )
  # grpc-install is on CMAKE_PREFIX_PATH and ships OpenSSL 1.x static libs, while
  # /usr/include is OpenSSL 3.x. rdkafka/hiredis then compile against 3.x headers but
  # propagate the 1.x libs — final links miss OSSL_*. Pin system OpenSSL like brpc.
  if(NOT SYSTEM_OPENSSL_SSL_LIB)
    find_library(SYSTEM_OPENSSL_SSL_LIB NAMES ssl
      PATHS /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib NO_DEFAULT_PATH)
  endif()
  if(NOT SYSTEM_OPENSSL_CRYPTO_LIB)
    find_library(SYSTEM_OPENSSL_CRYPTO_LIB NAMES crypto
      PATHS /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib NO_DEFAULT_PATH)
  endif()
  target_link_libraries(hps_native_support PUBLIC
    pthread
    rt
    base
    TBB::tbb
    rocksdb-shared
    redis++
    rdkafka
    ${SYSTEM_OPENSSL_SSL_LIB}
    ${SYSTEM_OPENSSL_CRYPTO_LIB}
    ${CUDA_CUDART_LIBRARY}
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
