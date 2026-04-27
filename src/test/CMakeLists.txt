find_package(folly REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_subdirectory(small_tests)
add_subdirectory(server_mgr)

# add_executable(test_f14_slab
#     test_f14_slab.cpp 
# )

# target_link_libraries(test_f14_slab PUBLIC
#     ${FOLLY_LIBRARIES}
#     mimalloc-static
# )

# target_compile_options(test_f14_slab PUBLIC -msse -msse2 -msse3 -msse4)

add_executable(test_hash_map
    test_hash_map.cpp 
)

target_link_libraries(test_hash_map PUBLIC
    ${FOLLY_LIBRARIES}
)

target_compile_options(test_hash_map PUBLIC -fopenmp)

target_link_libraries(test_hash_map PUBLIC -fopenmp)


add_executable(corotine 
    corotine.cpp
)

target_link_libraries(corotine PUBLIC
    ${BOOST_LIBRARIES}
    ${FOLLY_LIBRARIES}
)

add_executable(test_extendible_hash test_extendible_hash.cpp)
target_link_libraries(test_extendible_hash PUBLIC
    extendible_hash 
    base 
    gtest 
    gtest_main 
    pthread
    base_memory
)
add_test(NAME test_extendible_hash COMMAND test_extendible_hash)

add_executable(test_engine_extendible_hash test_engine_extendible_hash.cpp)
target_link_libraries(test_engine_extendible_hash PUBLIC
    extendible_hash
    base
    base_memory
    ps_common
    kv_engine
    gtest
    gtest_main
    pthread
    TBB::tbb
)
add_test(NAME test_engine_extendible_hash COMMAND test_engine_extendible_hash)

add_executable(test_cceh test_cceh.cpp)
target_link_libraries(test_cceh PUBLIC
    cceh
    base
    gtest
    gtest_main
    pthread
)
add_test(NAME test_cceh COMMAND test_cceh)

add_executable(test_engine_cceh test_engine_cceh.cpp)
target_link_libraries(test_engine_cceh PUBLIC
    cceh
    base
    base_memory
    ps_common
    kv_engine
    gtest
    gtest_main
    pthread
)
add_test(NAME test_engine_cceh COMMAND test_engine_cceh)

add_executable(test_io_backend test_io_backend.cpp)
target_link_libraries(test_io_backend PUBLIC
    cceh
    base
    gtest
    gtest_main
    pthread
)
add_test(NAME test_io_backend COMMAND test_io_backend)

add_executable(test_io_uring_test_utils test_io_uring_test_utils.cpp)
target_link_libraries(test_io_uring_test_utils PUBLIC
    gtest
    gtest_main
    uring
    pthread
)
add_test(NAME test_io_uring_test_utils COMMAND test_io_uring_test_utils)

add_executable(test_factory test_factory.cpp)
target_link_libraries(test_factory PUBLIC
    extendible_hash
    base
    base_memory
    ps_common
    kv_engine
    gtest
    gtest_main
    pthread
    TBB::tbb
)
add_test(NAME test_factory COMMAND test_factory)

add_executable(test_index test_index.cpp)
target_link_libraries(test_index PUBLIC
    extendible_hash 
    base 
    gtest 
    gtest_main 
    pthread
)
add_test(NAME test_index COMMAND test_index)

add_executable(test_ps_client_factory test_ps_client_factory.cpp)
target_sources(test_ps_client_factory PRIVATE
    ${PROJECT_SOURCE_DIR}/src/framework/common/ps_client_config_adapter.cc
    ${PROJECT_SOURCE_DIR}/src/ps/client_factory.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rdma_ps_client_adapter.cc
)
target_link_libraries(test_ps_client_factory PUBLIC
    base
    wq_ps_client
    grpc_ps_client
    brpc_ps_client
    local_shm_ps_client
    gflags
    glog
    gtest
    gtest_main
    pthread
)
target_link_options(test_ps_client_factory PUBLIC -Wl,--no-as-needed)
add_test(NAME test_ps_client_factory COMMAND test_ps_client_factory)

add_executable(test_local_shm_protocol test_local_shm_protocol.cpp)
target_link_libraries(test_local_shm_protocol PUBLIC
    base
    gtest
    gtest_main
    pthread
)
add_test(NAME test_local_shm_protocol COMMAND test_local_shm_protocol)

add_executable(test_local_shm_region test_local_shm_region.cpp)
target_link_libraries(test_local_shm_region PUBLIC
    base
    local_shm_common
    gtest
    gtest_main
    pthread
)
add_test(NAME test_local_shm_region COMMAND test_local_shm_region)

add_executable(test_local_shm_futex test_local_shm_futex.cpp)
target_link_libraries(test_local_shm_futex PUBLIC
    local_shm_common
    gtest
    gtest_main
    pthread
)
add_test(NAME test_local_shm_futex COMMAND test_local_shm_futex)

add_executable(test_local_shm_queue test_local_shm_queue.cpp)
target_link_libraries(test_local_shm_queue PUBLIC
    local_shm_common
    gtest
    gtest_main
    pthread
)
add_test(NAME test_local_shm_queue COMMAND test_local_shm_queue)

add_executable(test_local_shm_ps_client test_local_shm_ps_client.cpp)
target_link_libraries(test_local_shm_ps_client PUBLIC
    base
    local_shm_ps_client
    local_shm_ps_server_runtime
    gtest
    gtest_main
    pthread
)
add_test(NAME test_local_shm_ps_client COMMAND test_local_shm_ps_client)
set_tests_properties(test_local_shm_ps_client PROPERTIES
    TIMEOUT 30
    RUN_SERIAL TRUE
    RESOURCE_LOCK local_shm_tests
)

add_executable(test_op_runtime_support test_op_runtime_support.cpp)
target_sources(test_op_runtime_support PRIVATE
    ${PROJECT_SOURCE_DIR}/src/framework/common/op_runtime_support.cc
    ${PROJECT_SOURCE_DIR}/src/framework/common/ps_client_config_adapter.cc
    ${PROJECT_SOURCE_DIR}/src/ps/client_factory.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rdma_ps_client_adapter.cc
)
target_link_libraries(test_op_runtime_support PUBLIC
    base
    wq_ps_client
    grpc_ps_client
    brpc_ps_client
    local_shm_ps_client
    gflags
    glog
    gtest
    gtest_main
    pthread
)
add_test(NAME test_op_runtime_support COMMAND test_op_runtime_support)

add_executable(test_local_shm_op_component test_local_shm_op_component.cpp)
target_sources(test_local_shm_op_component PRIVATE
    ${PROJECT_SOURCE_DIR}/src/framework/common/local_shm_op_component.cc
)
target_link_libraries(test_local_shm_op_component PUBLIC
    base
    local_shm_ps_client
    local_shm_ps_server_runtime
    gtest
    gtest_main
    pthread
)
add_test(NAME test_local_shm_op_component COMMAND test_local_shm_op_component)
set_tests_properties(test_local_shm_op_component PROPERTIES
    TIMEOUT 30
    RUN_SERIAL TRUE
    RESOURCE_LOCK local_shm_tests
)

add_test(
    NAME test_run_local_shm_mixed_benchmark_py
    COMMAND ${Python3_EXECUTABLE} -m unittest
            src/test/scripts/test_run_local_shm_mixed_benchmark.py
)
set_tests_properties(
    test_run_local_shm_mixed_benchmark_py
    PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    LABELS "local_shm;python"
)

add_test(
    NAME test_run_local_shm_multi_process_benchmark_py
    COMMAND ${Python3_EXECUTABLE} -m unittest
            src/test/scripts/test_run_local_shm_multi_process_benchmark.py
)
set_tests_properties(
    test_run_local_shm_multi_process_benchmark_py
    PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    LABELS "local_shm;python"
)

add_executable(brpc_ps_server_unit_test
    test_brpc_ps_server.cpp
    ${PROJECT_SOURCE_DIR}/src/ps/brpc/brpc_ps_server.cpp
)
target_include_directories(brpc_ps_server_unit_test PUBLIC ${BRPC_INCLUDE_PATH})
target_include_directories(brpc_ps_server_unit_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/ps/brpc
)
target_compile_definitions(brpc_ps_server_unit_test PRIVATE RECSTORE_NO_SERVER_MAIN)
target_link_libraries(brpc_ps_server_unit_test PUBLIC
    base
    ps_base
    ps_proto
    ${BRPC_LIB}
    ${CUSTOM_PROTOBUF_LIB}
    ${SYSTEM_OPENSSL_SSL_LIB}
    ${SYSTEM_OPENSSL_CRYPTO_LIB}
    ZLIB::ZLIB
    Threads::Threads
    dl
    stdc++fs
    kv_engine
    sparse_tensor
    gflags
    leveldb
    TBB::tbb
    gtest
    gtest_main
)
add_test(NAME brpc_ps_server_unit_test COMMAND brpc_ps_server_unit_test)
set_tests_properties(brpc_ps_server_unit_test PROPERTIES LABELS "ps_unit")

add_executable(test_rdma_protocol test_rdma_protocol.cpp)
target_link_libraries(test_rdma_protocol PUBLIC
    base
    gtest
    gtest_main
    pthread
)
add_test(NAME test_rdma_protocol COMMAND test_rdma_protocol)

add_executable(test_allshards_ps_client test_allshards_ps_client.cpp)
target_link_libraries(test_allshards_ps_client PUBLIC
    base
    wq_ps_client
    gtest
    gtest_main
    pthread
)
add_test(NAME test_allshards_ps_client COMMAND test_allshards_ps_client)

add_executable(test_ps_transport_benchmark test_ps_transport_benchmark.cpp)
target_link_libraries(test_ps_transport_benchmark PUBLIC
    base
    gtest
    gtest_main
    pthread
)
add_test(NAME test_ps_transport_benchmark COMMAND test_ps_transport_benchmark)
add_executable(test_storage_cache_lru test_storage_cache_lru.cpp)
target_link_libraries(test_storage_cache_lru PUBLIC
    gtest
    gtest_main
    pthread
    storage_cache_lru
)
add_test(NAME test_storage_cache_lru COMMAND test_storage_cache_lru)

# report 单元测试
if(CMAKE_BUILD_TYPE MATCHES "Debug" AND USE_PERF_REPORT)
    add_executable(test_report test_report.cpp)
    target_link_libraries(test_report PUBLIC
        report_client
        gtest
        gtest_main
        pthread
    )
    add_test(NAME test_report COMMAND test_report)
endif()
