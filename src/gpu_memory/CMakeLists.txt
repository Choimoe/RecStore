add_library(gpu_memory memory_pool.cu.cc memory_pool.h)

find_package(rmm REQUIRED)

target_link_libraries(
  gpu_memory 
  PUBLIC
  rmm::rmm
  ${CMAKE_CURRENT_SOURCE_DIR}/gdrcopy/src/libgdrapi.so
)

add_executable(test_gdr_size
test_gdr_size.cu
)
target_link_libraries(
  test_gdr_size
  gpu_memory
  cuda
)
set_target_properties(test_gdr_size PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(test_gdr_size PROPERTIES CUDA_ARCHITECTURES OFF)
