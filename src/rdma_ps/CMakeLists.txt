add_library(wq_ps_client petps_client.cc)
                         
target_link_libraries(wq_ps_client base mayfly)
set(warp_wq_ps_client -Wl,--whole-archive wq_ps_client -Wl,--no-whole-archive CACHE INTERNAL "wq_ps_client")

add_executable(petps_server petps_server.cc)
target_link_libraries(petps_server PUBLIC base base_ps kv_engine ps_common mayfly TBB::tbb )
