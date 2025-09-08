set(CITYHASH_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/cityhash)
include(${PROJECT_SOURCE_DIR}/cmake/FindCityhash.cmake)

find_package(cpptrace REQUIRED)

add_library(base base.cc pprint.cc shm_helper.cc bind_core.cc counter.cc)
target_link_libraries(base PUBLIC Folly::folly ${CITYHASH_LIBRARIES} gflags glog)
target_include_directories(base PUBLIC ${CITYHASH_INCLUDE_DIRS})
target_link_libraries(base PUBLIC protobuf::libprotobuf protobuf::libprotobuf-lite nlohmann_json::nlohmann_json)
target_link_libraries(base PUBLIC cpptrace::cpptrace)


add_executable(pq_test test_pq.cc)
target_link_libraries(pq_test PUBLIC base)



find_package(pybind11 REQUIRED)
pybind11_add_module(timer_module MODULE pybind_timer.cc)
target_link_libraries(timer_module PRIVATE base)