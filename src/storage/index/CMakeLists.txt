add_subdirectory(dram)
add_subdirectory(ssd)

add_library(index INTERFACE)

target_include_directories(index
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(index
  INTERFACE
    extendible_hash
    cceh
)
