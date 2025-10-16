find_package(folly REQUIRED)

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

# ExtendibleHash 单元测试
add_executable(test_extendible_hash test_extendible_hash.cpp)
target_link_libraries(test_extendible_hash PUBLIC
    dram_hash 
    base 
    gtest 
    gtest_main 
    pthread
    base_memory
)
add_test(NAME test_extendible_hash COMMAND test_extendible_hash)

# SSDExtendibleHash 单元测试
add_executable(test_ssd_extendible_hash test_ssd_extendible_hash.cpp)
target_link_libraries(test_ssd_extendible_hash PUBLIC
    dram_hash 
    base 
    gtest 
    gtest_main 
    pthread
    base_memory
    cceh
)

# Engine Extendible Hash 测试
add_executable(test_engine_extendible_hash test_engine_extendible_hash.cpp)
target_link_libraries(test_engine_extendible_hash PUBLIC
    dram_hash
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

# CCEH 测试
add_executable(test_cceh test_cceh.cpp)
target_link_libraries(test_cceh PUBLIC
    cceh
    base
    gtest
    gtest_main
    pthread
)

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

add_executable(test_factory test_factory.cpp)
target_link_libraries(test_factory PUBLIC
    dram_hash
    base
    base_memory
    ps_common
    kv_engine
    gtest
    gtest_main
    pthread
    TBB::tbb
)

# index 单元测试
add_executable(test_index test_index.cpp)
target_link_libraries(test_index PUBLIC
    dram_hash 
    base 
    gtest 
    gtest_main 
    pthread
)
