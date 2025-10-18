add_library(base_ps parameters.cpp cache_ps_impl.h parameters.h)

add_library(ps_common Postoffice.cc config.cc)
target_link_libraries(ps_common PUBLIC base memcached )
target_include_directories(ps_common PUBLIC ${PROJECT_BINARY_DIR}/third_party/Mayfly-main)
