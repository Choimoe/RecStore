project(UVM_LIBTORCH LANGUAGES CXX CUDA C)
set(TORCH_CUDA_ARCH_LIST "8.6" CACHE STRING "CUDA architecture version")

find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)


add_executable(uvm_libtorch uvm_libtorch.cc)
target_include_directories(uvm_libtorch PUBLIC
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)
target_link_libraries(uvm_libtorch PUBLIC
    base
    ${TORCH_LIBRARIES}
    Python3::Python
)
target_link_libraries(uvm_libtorch PRIVATE Python3::Python)
target_link_libraries(uvm_libtorch PUBLIC base)

set_target_properties(uvm_libtorch PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(uvm_libtorch PROPERTIES CUDA_ARCHITECTURES OFF)



add_library(intercept SHARED intercept.cpp)
target_link_libraries(intercept PUBLIC dl /usr/local/cuda/lib64/libcudart.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so)
target_include_directories(intercept PUBLIC /usr/local/cuda/include)
set_target_properties(intercept PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(intercept PROPERTIES CUDA_ARCHITECTURES OFF)


# g++ -I/usr/local/cuda/include -fPIC -shared -o intercept.so intercept.cpp -ldl -L/usr/local/cuda/lib64 -lcudart -lcuda