find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)

add_subdirectory(proto)


add_library(grpc_ps_client
    grpc_ps_client.cpp grpc_ps_client.h
)

find_library(REPORT_LIB report PATHS ${CMAKE_CURRENT_SOURCE_DIR} NO_DEFAULT_PATH)
if(NOT REPORT_LIB)
    message(WARNING "libreport.so not found in ${CMAKE_CURRENT_SOURCE_DIR}, performance reporting will not work")
    set(REPORT_LIB "")
endif()

target_link_libraries(
    grpc_ps_client PUBLIC
    base
    base_ps
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    protobuf::libprotobuf
    ${REPORT_LIB}
)

add_executable(grpc_ps_client_test
    grpc_ps_client_test.cpp
)
target_link_libraries(grpc_ps_client_test PUBLIC
    grpc_ps_client
)


add_executable(grpc_ps_server
    grpc_ps_server.cpp
    grpc_ps_server.h
)

target_link_libraries(grpc_ps_server PUBLIC
    base
    base_ps
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    protobuf::libprotobuf
    stdc++fs
    kv_engine
    gflags
    glog
    TBB::tbb
    ${REPORT_LIB}
)

add_library(dist_grpc_ps_client
    dist_grpc_ps_client.cpp dist_grpc_ps_client.h
)
target_link_libraries(dist_grpc_ps_client PUBLIC
    base
    base_ps
    grpc_ps_client
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    protobuf::libprotobuf
    ${REPORT_LIB}
)

add_executable(dist_grpc_ps_client_test
    dist_grpc_ps_client_test.cpp
)
target_link_libraries(dist_grpc_ps_client_test PUBLIC
    dist_grpc_ps_client
)

# add_executable(grpc_ps_server_async
#     grpc_ps_server_async.cpp 
#     grpc_ps_server_async.h 
#     cache_ps_impl.h
#     ${common}
# )

# target_link_libraries(grpc_ps_server_async PUBLIC
#     base
#     ps_proto
#     gRPC::grpc++_reflection
#     gRPC::grpc++
#     protobuf::libprotobuf
#     stdc++fs
#     kv_engine
# )


add_library(grpc_ps_client_python MODULE
    grpc_ps_client_pytorch.cpp
)

target_include_directories(grpc_ps_client_python PUBLIC ${TORCH_INCLUDE_DIRS})

target_link_libraries(grpc_ps_client_python PUBLIC
    grpc_ps_client
    ${TORCH_LIBRARIES}
    Python3::Python
)
