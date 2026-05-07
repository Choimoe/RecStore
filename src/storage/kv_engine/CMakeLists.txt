find_package(PkgConfig QUIET)

option(USE_PMKV "enable PMKV" OFF)
if(USE_PMKV MATCHES "ON")
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(LIBPMEMKV REQUIRED libpmemkv)
  else()
    message(FATAL_ERROR "pkg-config not found")
  endif()

  include_directories(${LIBPMEMKV_INCLUDE_DIRS})
  link_directories(${LIBPMEMKV_LIBRARY_DIRS})
endif()

# KV engine factory
add_library(kv_engine INTERFACE)
target_link_libraries(kv_engine INTERFACE base)
if(USE_PMKV MATCHES "ON")
  target_link_libraries(kv_engine INTERFACE ${LIBPMEMKV_LIBRARIES})
endif()

target_link_libraries(
  kv_engine
  INTERFACE pet_kv index io_backend
  # spdk_wrapper
)
  # INTERFACE pet_kv dash_hash PMHashPCLHT PMHashLevel PMHashClevel PMHashCCEHVM)

cmake_policy(SET CMP0076 NEW)

target_sources(
  kv_engine INTERFACE kv_engine_register.cc)
# KV engine factory done

add_executable(benchmark_kv_engine benchmark_kv_engine.cc)
target_link_libraries(benchmark_kv_engine PUBLIC pthread base kv_engine ps_common)
target_link_libraries(benchmark_kv_engine PUBLIC OpenMP::OpenMP_CXX)

add_executable(hps_backend_benchmark
  hps_backend_benchmark.cc
  hps_backend_shim.cc
  ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/core23/logger.cpp
  ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/thread_pool.cpp
  ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/hps/database_backend.cpp
  ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/src/hps/hash_map_backend.cpp
)
target_include_directories(hps_backend_benchmark PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR
  ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/include
  ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/parallel-hashmap
  ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/rocksdb/include
  ${CUDA_TOOLKIT_INCLUDE}
)
target_link_libraries(hps_backend_benchmark PUBLIC pthread base kv_engine ps_common TBB::tbb)
target_link_libraries(hps_backend_benchmark PUBLIC OpenMP::OpenMP_CXX)

add_executable(bench benchmark_zipf.cc)
target_link_libraries(bench PUBLIC pthread base kv_engine)
