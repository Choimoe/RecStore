# Framework-neutral cache: no Torch or Python include/link dependencies.
add_library(framework_gpu_cache STATIC gpu_embedding_cache.cu)
target_compile_definitions(framework_gpu_cache PUBLIC RECSTORE_ENABLE_GPU_CACHE)
target_include_directories(framework_gpu_cache PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(framework_gpu_cache PUBLIC gpu_cache glog::glog)
target_compile_options(framework_gpu_cache PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(framework_gpu_cache PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES OFF
    POSITION_INDEPENDENT_CODE ON)

# This executable exercises the complete cache API without linking Torch.
add_executable(test_framework_gpu_cache ${PROJECT_SOURCE_DIR}/src/test/test_framework_gpu_cache.cu)
target_link_libraries(test_framework_gpu_cache PRIVATE framework_gpu_cache gtest gtest_main pthread)
set_target_properties(test_framework_gpu_cache PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME test_framework_gpu_cache COMMAND test_framework_gpu_cache)
set_tests_properties(test_framework_gpu_cache PROPERTIES TIMEOUT 60)
