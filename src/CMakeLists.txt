set(RECSTORE_PATH ${PROJECT_SOURCE_DIR})
configure_file(${PROJECT_SOURCE_DIR}/src/recstore_config.h.in
	${PROJECT_BINARY_DIR}/recstore_config.h
)
include_directories(${PROJECT_BINARY_DIR})


add_subdirectory(base)

add_subdirectory(ps)

add_subdirectory(memory)
add_subdirectory(storage)
add_subdirectory(optimizer)
if(RECSTORE_BUILD_BENCHMARKS)
	add_subdirectory(benchmark)
else()
	message(STATUS "Skipping RecStore benchmarks")
endif()

if(RECSTORE_BUILD_TESTS)
	add_subdirectory(test)
else()
	message(STATUS "Skipping RecStore tests")
endif()

if(RECSTORE_BUILD_FRAMEWORK)
	add_subdirectory(framework)
else()
	message(STATUS "Skipping RecStore framework bindings")
endif()

if(RECSTORE_BUILD_UVM_EXECUTABLES AND ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
	add_subdirectory(executable/uvm)
else()
	message(STATUS "Skipping executable/uvm because CUDA is disabled or nvcc not found")
endif()
