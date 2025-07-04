set(TORCH_CUDA_ARCH_LIST "8.6" CACHE STRING "CUDA architecture version")
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(GRAD_ASYNC_V1_DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    # add_compile_definitions(GRAD_ASYNC_V1_DEBUG)
else()
    message(FATAL_ERROR "CMAKE_BUILD_TYPE must be Debug or Release")
endif()

cmake_host_system_information(RESULT xmh_host_name  QUERY HOSTNAME)

if(xmh_host_name STREQUAL "node182")
    add_compile_definitions(DEFINED_SHM_GB=150)
elseif(xmh_host_name STREQUAL "3090-server")
    add_compile_definitions(DEFINED_SHM_GB=200)
else()
    message(WARNING "Unknown machine: ${xmh_host_name}, using default SHM size: 50GB")
    add_compile_definitions(DEFINED_SHM_GB=50)
endif()


set(recstore_pytorch_srcs
    recstore_pytorch.cu 
    recstore_pytorch.cc
    merge.cu 
    kg_controller.cc
    IPCTensor.cc
    IPC_barrier.cc
    torch_utils.cc
    parallel_pq.cc
    grad_async_v1.h
    grad_async_v2.h
)

add_library(recstore_pytorch MODULE 
    ${recstore_pytorch_srcs}
)

target_include_directories(recstore_pytorch PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(recstore_pytorch PUBLIC ${TORCH_LIBRARIES} )
target_link_libraries(recstore_pytorch PUBLIC base gpu_cache TBB::tbb)
target_link_libraries(recstore_pytorch PRIVATE Python3::Python)

set_target_properties(recstore_pytorch PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(recstore_pytorch PROPERTIES CUDA_ARCHITECTURES OFF)



add_executable(recstore_pytorch_test 
    kg_controller_test.cc
    ${recstore_pytorch_srcs}
)
target_include_directories(recstore_pytorch_test PUBLIC ${TORCH_INCLUDE_DIRS})

target_link_libraries(recstore_pytorch_test PUBLIC 
${TORCH_LIBRARIES} base Python3::Python
gpu_cache)

set_target_properties(recstore_pytorch_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(recstore_pytorch_test PROPERTIES CUDA_ARCHITECTURES OFF)

# target_compile_options(recstore_pytorch_test PUBLIC "-fsanitize=thread")
# target_link_options(recstore_pytorch_test PUBLIC "-fsanitize=thread")