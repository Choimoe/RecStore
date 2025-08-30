# src/memory/CMakeLists.txt

add_library(base_memory
  persist_malloc.cc
  shm_file.cc
  allocator_master.hh
  allocator.hh
)

target_include_directories(base_memory PRIVATE
  ${CMAKE_SOURCE_DIR}/src/memory
  ${JEMALLOC_INSTALL_DIR}/include
)

add_dependencies(base_memory jemalloc)
target_link_libraries(base_memory PUBLIC base jemalloc)

add_executable(pm_command pm_command.cc)
target_link_libraries(pm_command PUBLIC base_memory)