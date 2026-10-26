add_library(ssd_layout ssd_layout.cc)
target_include_directories(ssd_layout PUBLIC ${PROJECT_SOURCE_DIR}/src)

add_executable(recstore_ssd_layout main.cc)
target_link_libraries(recstore_ssd_layout PRIVATE ssd_layout)
