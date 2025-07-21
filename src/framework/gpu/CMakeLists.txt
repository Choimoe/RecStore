# export TORCH_CUDA_ARCH_LIST="8.0;8.6"
# export CUDAARCHS="80;86"

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

find_package(Python 3.10 COMPONENTS Interpreter REQUIRED)

set(GPU_KV_STORE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_kv_store.cu
)

add_library(gpu_kv_store STATIC ${GPU_KV_STORE_SRC})
target_include_directories(gpu_kv_store PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)
target_include_directories(gpu_kv_store PUBLIC /usr/include/python3.10)
target_link_libraries(gpu_kv_store PUBLIC
    ${CUDA_LIBRARIES}
    ${TORCH_LIBRARIES}
)
set_target_properties(gpu_kv_store PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_test(
    NAME dist_emb_single_gpu_unittest
    COMMAND ${Python_EXECUTABLE} -m unittest recstore.unittest.test_dist_emb_single_gpu
)

set_tests_properties(dist_emb_single_gpu_unittest PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/pytorch
)