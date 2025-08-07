option(USE_FAKE_KVCLIENT "Use FakeKVClientOp for testing" OFF)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pytorch)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tensorflow)

if(USE_FAKE_KVCLIENT)
    message(STATUS "Using FakeKVClientOp for testing!")
    add_definitions(-DUSE_FAKE_KVCLIENT)
endif()
