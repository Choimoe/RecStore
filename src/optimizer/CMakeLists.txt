add_library(sparse_tensor
	sparse_tensor.cpp
	optimizer.cpp
)

target_link_libraries(sparse_tensor PUBLIC
    kv_engine
)

if(CMAKE_BUILD_TYPE MATCHES "Debug" AND USE_PERF_REPORT)
    target_compile_definitions(sparse_tensor PRIVATE ENABLE_PERF_REPORT)
    target_link_libraries(sparse_tensor PRIVATE report_client)
endif()
