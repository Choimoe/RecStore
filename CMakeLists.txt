cmake_minimum_required(VERSION 3.5)
project(RecStore LANGUAGES CXX CUDA C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CUDA_SEPARABLE_COMPILATION ON) 
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "") # works

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads)


if (OPENMP_FOUND)
    message(STATUS "OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fuse-ld=mold -faligned-new -Wno-unused-function")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -faligned-new -Wno-unused-function")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")


# setting compiler flags

set(SM "80;86")
foreach(arch_name ${SM})
    if (arch_name STREQUAL 86 OR 
        arch_name STREQUAL 80 OR 
        arch_name STREQUAL 75 OR 
        arch_name STREQUAL 70 OR 
        arch_name STREQUAL 61 OR 
        arch_name STREQUAL 60)
        list(APPEND cuda_arch_list ${arch_name})
        message(STATUS "-- Assign GPU architecture (sm=${arch_name})")
    else()
        message(FATAL_ERROR "-- Unknown or unsupported GPU architecture (set sm=70)")
    endif()
endforeach()

list(LENGTH cuda_arch_list cuda_arch_list_length)
if(${cuda_arch_list_length} EQUAL 0)
    list(APPEND cuda_arch_list "80")
endif()
list(REMOVE_DUPLICATES cuda_arch_list)

foreach(arch_name ${cuda_arch_list})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch_name},code=sm_${arch_name}")
endforeach()

set(CC_WARNING_FLAG "-Wall -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-value -Wno-unused-variable -Wno-attributes -Wno-parentheses -Wno-unused-but-set-variable")
string(REPLACE " " "," NVCC_WARNING_FLAG ${CC_WARNING_FLAG})

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -fPIC -g -ggdb3 -Wall")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC -g -ggdb3 ${CC_WARNING_FLAG} --std=c++17")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin /usr/bin/g++ -rdc=true --std=c++17 -Xcompiler ${NVCC_WARNING_FLAG},-g,-ggdb3,-fPIC -lineinfo --expt-extended-lambda --expt-relaxed-constexpr")

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG} -fPIC -g -ggdb3 -O0 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -fPIC -g -ggdb3 -O0 ${CC_WARNING_FLAG}")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}  -g -G -Xptxas -O0 -lineinfo -O0 -Xcompiler -O0,${NVCC_WARNING_FLAG},-g,-ggdb3,-fPIC")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fPIC -g -ggdb3 -O3 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -fPIC -g -ggdb3 -O3 ${CC_WARNING_FLAG}")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}  -g -Xptxas -O3 -lineinfo -O3 -Xcompiler -O3,${NVCC_WARNING_FLAG},-g,-ggdb3,-fPIC")


# setting output folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# configure_file(${PROJECT_SOURCE_DIR}/HugeCTR/include/config.hpp.in ${PROJECT_SOURCE_DIR}/HugeCTR/include/config.hpp)

# building

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/third_party/grpc-install")

include("${CMAKE_CURRENT_LIST_DIR}/cmake/FindGRPC.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/common.cmake")



list(APPEND CMAKE_PREFIX_PATH "/home/xieminhui/.local/share/cmake/")
list(APPEND CMAKE_PREFIX_PATH "/home/pod/.local/share/cmake/")

list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/third_party/libtorch/libtorch/share/cmake/Torch")
set(Torch_DIR "${PROJECT_SOURCE_DIR}/third_party/libtorch/libtorch/share/cmake/Torch")

# list(APPEND CMAKE_PREFIX_PATH "/home/xieminhui/.local/lib/python3.10/site-packages/torch/share/cmake/Torch/")


# if (EXISTS "/home/pod/.local/lib/python3.10/site-packages/torch/share/cmake/Torch")
#   list(APPEND CMAKE_PREFIX_PATH "/home/pod/.local/lib/python3.10/site-packages/torch/share/cmake/Torch")
# endif()

# if (EXISTS "/home/xieminhui/miniconda3/lib/python3.11/site-packages/torch/share/cmake/Torch")
#   set(Torch_DIR "/home/xieminhui/miniconda3/lib/python3.11/site-packages/torch/share/cmake/Torch")
# endif()

# if (EXISTS "/home/xieminhui/anaconda3/lib/python3.1/site-packages/torch/share/cmake/Torch")
#   set(Torch_DIR "/home/xieminhui/anaconda3/lib/python3.1/site-packages/torch/share/cmake/Torch")
# endif()







set(folly_DIR "${PROJECT_SOURCE_DIR}/third_party/folly/folly-install-fPIC/usr/local/lib/cmake/folly")
set(FOLLY_ROOT_DIR "${folly_DIR}")


find_package(folly REQUIRED)
find_package(gflags REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/third_party/folly/folly-install-fPIC/usr/local/include)
link_directories(${PROJECT_SOURCE_DIR}/third_party/folly/folly-install-fPIC/usr/local/lib)

set(glog_DIR "${PROJECT_SOURCE_DIR}/third_party/glog/glog-install-fPIC/usr/local/lib/cmake/glog")
find_package(glog CONFIG REQUIRED)


if(OPENMP_FOUND)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
  message(STATUS "add -fopenmp to compiler")
else()
  message(FATAL_ERROR "without openmp the multi-node all2all will fail")
endif()

add_subdirectory(third_party/googletest)
add_subdirectory(third_party/Mayfly-main)
add_subdirectory(third_party/json)
# add_subdirectory(third_party/pybind11)
# add_subdirectory(third_party/mimalloc)

set(CTEST_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/Testing")
enable_testing()

include_directories(
  ${PROJECT_SOURCE_DIR}/src
)
include_directories(
  ${PROJECT_SOURCE_DIR}
)


set(ONEAPI_ROOT /opt/intel/oneapi)
set(TBB_ROOT ${ONEAPI_ROOT}/tbb/latest/lib)
list(APPEND CMAKE_PREFIX_PATH ${TBB_ROOT}/cmake)
find_package(TBB REQUIRED)


message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
