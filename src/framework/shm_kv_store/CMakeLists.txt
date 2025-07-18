find_package(Torch REQUIRED)
add_library(recstore_shm_kv_store STATIC
    shm_kv_store.cc
)
target_include_directories(recstore_shm_kv_store PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(recstore_shm_kv_store PUBLIC
    rt
    ${BOOST_LIBRARIES}
    ${FOLLY_LIBRARIES}
    ${TORCH_LIBRARIES}
)

