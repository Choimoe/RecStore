add_library(dram_hash)

target_sources(dram_hash 
  PRIVATE 
   extendible_hash.cpp
)

target_include_directories(dram_hash
  PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(dram_hash
  PUBLIC
    base
)
