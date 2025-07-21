find_package(Torch REQUIRED)
find_package(MPI REQUIRED)

set(ENV{PKG_CONFIG_PATH} "/opt/ucx/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
set(ENV{CPATH} "/opt/ucx/include:$ENV{CPATH}")
set(ENV{LD_LIBRARY_PATH} "/opt/ucx/lib:$ENV{LD_LIBRARY_PATH}")

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
    MPI::MPI_CXX
)


find_package(Python 3.10 COMPONENTS Interpreter REQUIRED)
find_package(MPI REQUIRED)

add_test(
    NAME dist_emb_shm_unittest
    COMMAND ${MPIEXEC_EXECUTABLE} -np 2 ${Python_EXECUTABLE} -m unittest recstore.unittest.test_dist_emb_shm
)

set_tests_properties(dist_emb_shm_unittest PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/pytorch
)

