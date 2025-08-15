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

#hybridkv 单元测试
add_executable(test_hybridkv test_hybridkv.cpp)
target_link_libraries(test_hybridkv PUBLIC
    TBB::tbb
    dram_hash 
    base 
    gtest 
    gtest_main 
    pthread
    base_memory
)