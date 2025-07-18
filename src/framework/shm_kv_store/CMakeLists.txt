find_package(Torch REQUIRED)

set(UCC_ROOT ${PROJECT_SOURCE_DIR}/third_party/ucc/build)
set(UCC_INCLUDE_DIR ${UCC_ROOT}/include)
set(UCC_LIBRARY_DIR ${UCC_ROOT}/lib)

add_library(recstore_shm_kv_store STATIC
    shm_kv_store.cc
    ucc_context.cc
)

target_include_directories(recstore_shm_kv_store PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${TORCH_INCLUDE_DIRS}
    ${UCC_INCLUDE_DIR}
)

target_link_directories(recstore_shm_kv_store PUBLIC
    ${UCC_LIBRARY_DIR}
)

target_link_libraries(recstore_shm_kv_store PUBLIC
    rt
    ${BOOST_LIBRARIES}
    ${FOLLY_LIBRARIES}
    ${TORCH_LIBRARIES}
    ucc
)

