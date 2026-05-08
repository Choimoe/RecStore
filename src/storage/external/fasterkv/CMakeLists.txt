set(RECSTORE_FASTER_DIR ${PROJECT_SOURCE_DIR}/third_party/faster/cc)

if(EXISTS "${RECSTORE_FASTER_DIR}/src/CMakeLists.txt" AND NOT TARGET faster)
  add_library(faster STATIC
    ${RECSTORE_FASTER_DIR}/src/core/address.cc
    ${RECSTORE_FASTER_DIR}/src/core/lss_allocator.cc
    ${RECSTORE_FASTER_DIR}/src/core/thread.cc
    ${RECSTORE_FASTER_DIR}/src/environment/file_linux.cc
  )
  target_include_directories(faster PUBLIC
    ${RECSTORE_FASTER_DIR}/src
  )
  target_link_libraries(faster PUBLIC
    TBB::tbb
    uuid
    aio
    pthread
  )
endif()

if(TARGET faster)
  add_library(fasterkv_backend STATIC
    fasterkv_backend.cc
  )
  target_include_directories(fasterkv_backend PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${RECSTORE_FASTER_DIR}/src
  )
  target_link_libraries(fasterkv_backend PUBLIC
    faster
    TBB::tbb
    pthread
  )
else()
  message(STATUS
    "Skipping fasterkv_backend: third_party/faster/cc is unavailable."
  )
endif()
