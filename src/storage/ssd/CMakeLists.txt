add_library(cceh)

target_sources(cceh 
  PRIVATE 
    CCEH.cpp
    ssd_extendable_hash.cpp
)

target_include_directories(cceh
  PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cceh
  PUBLIC
    base
  PRIVATE
    Threads::Threads
    uring
)
