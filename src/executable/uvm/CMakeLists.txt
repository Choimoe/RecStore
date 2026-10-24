find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)

add_executable(uvm_libtorch uvm_libtorch.cc)
target_link_libraries(uvm_libtorch PRIVATE
    base
    ${TORCH_LIBRARIES}
    Python3::Python
)
set_target_properties(uvm_libtorch PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_library(intercept SHARED intercept.cpp)
target_include_directories(intercept PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(intercept PRIVATE
    ${CMAKE_DL_LIBS}
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_CUDA_LIBRARY}
)
set_target_properties(intercept PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
