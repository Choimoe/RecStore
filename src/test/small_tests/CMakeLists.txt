if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
	add_subdirectory(perf_uva)
else()
	message(STATUS "Skipping small_tests/perf_uva because CUDA is disabled or nvcc not found")
endif()


add_executable(test_multi_thread_cht test_multi_thread_cht.cc)
target_link_libraries(test_multi_thread_cht base)


# add_executable(test_print test_print.cu)
