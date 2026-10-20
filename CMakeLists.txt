cmake_minimum_required(VERSION 3.10)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING
    "Minimum policy version accepted from third-party CMake projects")

option(ENABLE_CUDA "Build with CUDA support" ON)
option(USE_BRPC_CPU_PROFILER "Enable bRPC builtin CPU profiler" OFF)

if(ENABLE_CUDA AND NOT DEFINED CMAKE_CUDA_COMPILER)
  set(_recstore_nvcc_candidates
    /usr/local/cuda/bin/nvcc
    /usr/bin/nvcc
  )
  file(GLOB _recstore_nvcc_glob_candidates "/usr/local/cuda-*/bin/nvcc")
  list(APPEND _recstore_nvcc_candidates ${_recstore_nvcc_glob_candidates})

  foreach(_recstore_nvcc IN LISTS _recstore_nvcc_candidates)
    if(EXISTS "${_recstore_nvcc}")
      set(CMAKE_CUDA_COMPILER "${_recstore_nvcc}" CACHE FILEPATH "CUDA compiler" FORCE)
      get_filename_component(_recstore_cuda_bin "${_recstore_nvcc}" DIRECTORY)
      get_filename_component(_recstore_cuda_root "${_recstore_cuda_bin}" DIRECTORY)
      set(CUDAToolkit_ROOT "${_recstore_cuda_root}" CACHE PATH "CUDA toolkit root" FORCE)
      set(CUDA_TOOLKIT_ROOT_DIR "${_recstore_cuda_root}" CACHE PATH "CUDA toolkit root (FindCUDA compatibility)" FORCE)
      message(STATUS "Auto-detected nvcc: ${CMAKE_CUDA_COMPILER}")
      break()
    endif()
  endforeach()
endif()

# Single source of truth for the version: the repository-root VERSION file is
# read here for the CMake build and by pyproject.toml for the package metadata.
# It may carry a PEP 440 local segment naming the build variant (e.g.
# "0.1.0+cu126", the same convention as torch's "2.7.1+cu126" wheels); that
# segment shows up in the wheel filename and in recstore.__version__.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" RECSTORE_VERSION_RAW)
string(STRIP "${RECSTORE_VERSION_RAW}" RECSTORE_VERSION)

# project(VERSION) only accepts numeric components, so drop the local segment
# for CMake while RECSTORE_VERSION keeps the full string for the package.
string(REGEX REPLACE "\\+.*$" "" RECSTORE_VERSION_NUMERIC "${RECSTORE_VERSION}")
if(NOT RECSTORE_VERSION_NUMERIC MATCHES "^[0-9]+\\.[0-9]+")
  message(FATAL_ERROR
    "VERSION must start with a numeric release (MAJOR.MINOR[.PATCH]); got '${RECSTORE_VERSION}'")
endif()

if(ENABLE_CUDA)
  project(RecStore VERSION ${RECSTORE_VERSION_NUMERIC} LANGUAGES CXX CUDA C)
else()
  project(RecStore VERSION ${RECSTORE_VERSION_NUMERIC} LANGUAGES CXX C)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)
if(ENABLE_CUDA)
  set(CUDA_SEPARABLE_COMPILATION ON)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "") # works

if(ENABLE_CUDA)
  find_package(CUDA REQUIRED)
  set(RECSTORE_CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} ${CUDA_TOOLKIT_INCLUDE})
  find_path(RECSTORE_CUBLAS_INCLUDE_DIR cublas_v2.h
    HINTS
      ${RECSTORE_CUDA_INCLUDE_DIRS}
      ${CUDAToolkit_ROOT}/include
      ${CUDAToolkit_ROOT}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/include
      ${CUDA_TOOLKIT_ROOT_DIR}/include
      ${CUDA_TOOLKIT_ROOT_DIR}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/include
  )
  find_path(RECSTORE_CURAND_INCLUDE_DIR curand.h
    HINTS ${RECSTORE_CUDA_INCLUDE_DIRS} ${RECSTORE_CUBLAS_INCLUDE_DIR}
  )
  find_path(RECSTORE_NVML_INCLUDE_DIR nvml.h
    HINTS ${RECSTORE_CUDA_INCLUDE_DIRS} ${RECSTORE_CUBLAS_INCLUDE_DIR}
  )
  if(RECSTORE_CUBLAS_INCLUDE_DIR AND RECSTORE_CURAND_INCLUDE_DIR AND
     RECSTORE_NVML_INCLUDE_DIR)
    list(APPEND RECSTORE_CUDA_INCLUDE_DIRS
      ${RECSTORE_CUBLAS_INCLUDE_DIR}
      ${RECSTORE_CURAND_INCLUDE_DIR}
      ${RECSTORE_NVML_INCLUDE_DIR}
    )
    list(REMOVE_DUPLICATES RECSTORE_CUDA_INCLUDE_DIRS)
    set(RECSTORE_HUGECTR_CUDA_HEADERS_AVAILABLE ON)
  else()
    set(RECSTORE_HUGECTR_CUDA_HEADERS_AVAILABLE OFF)
    message(STATUS
      "HugeCTR CUDA headers are incomplete; targets including HugeCTR common.hpp will be skipped."
    )
  endif()
endif()
find_package(OpenMP REQUIRED)
find_package(Threads)

include(ExternalProject)




set(JEMALLOC_INSTALL_DIR ${CMAKE_BINARY_DIR}/jemalloc-install)

ExternalProject_Add(jemalloc_ext
  SOURCE_DIR        ${CMAKE_SOURCE_DIR}/third_party/jemalloc   # 你的 jemalloc 源码目录
  CONFIGURE_COMMAND autoconf
  BUILD_IN_SOURCE   1
  BUILD_COMMAND     ./configure
                      --with-jemalloc-prefix=je
                      --with-pic
                      --prefix=${JEMALLOC_INSTALL_DIR}
                      CC=${CMAKE_C_COMPILER}
                      CXX=${CMAKE_CXX_COMPILER}
                      CFLAGS=-fPIC
                      CXXFLAGS=-fPIC
                    && $(MAKE) -j
  INSTALL_COMMAND   $(MAKE) install
)


set(JEMALLOC_LIB_PATH ${JEMALLOC_INSTALL_DIR}/lib/libjemalloc.a)


file(MAKE_DIRECTORY ${JEMALLOC_INSTALL_DIR}/include)
file(MAKE_DIRECTORY ${JEMALLOC_INSTALL_DIR}/lib)

add_library(jemalloc STATIC IMPORTED GLOBAL)   
set_target_properties(jemalloc PROPERTIES
  IMPORTED_LOCATION             ${JEMALLOC_LIB_PATH}
  INTERFACE_INCLUDE_DIRECTORIES ${JEMALLOC_INSTALL_DIR}/include
)
add_dependencies(jemalloc jemalloc_ext)


list(APPEND CMAKE_BUILD_RPATH   "${JEMALLOC_INSTALL_DIR}/lib")
list(APPEND CMAKE_INSTALL_RPATH "${JEMALLOC_INSTALL_DIR}/lib")


if (OPENMP_FOUND)
    message(STATUS "OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fuse-ld=mold -faligned-new -Wno-unused-function")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -faligned-new -Wno-unused-function")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")


# setting compiler flags

if(ENABLE_CUDA)
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

  set(torch_cuda_arch_list)
  foreach(arch_name ${cuda_arch_list})
    string(SUBSTRING "${arch_name}" 0 1 arch_major)
    string(SUBSTRING "${arch_name}" 1 1 arch_minor)
    list(APPEND torch_cuda_arch_list "${arch_major}.${arch_minor}")
  endforeach()
  set(TORCH_CUDA_ARCH_LIST "${torch_cuda_arch_list}" CACHE STRING "CUDA architectures used by Torch CMake targets" FORCE)
endif()

set(CC_WARNING_FLAG "-Wall -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-value -Wno-unused-variable -Wno-attributes -Wno-parentheses -Wno-unused-but-set-variable")
string(REPLACE " " "," NVCC_WARNING_FLAG ${CC_WARNING_FLAG})

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -fPIC -g -ggdb3 -Wall")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC -g -ggdb3 ${CC_WARNING_FLAG} --std=c++17")
if(ENABLE_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin /usr/bin/g++ -rdc=true --std=c++17 -Xcompiler ${NVCC_WARNING_FLAG},-g,-ggdb3,-fPIC -lineinfo --expt-extended-lambda --expt-relaxed-constexpr")
endif()

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG} -fPIC -g -ggdb3 -O0 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -fPIC -g -ggdb3 -O0 ${CC_WARNING_FLAG}")
if(ENABLE_CUDA)
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}  -g -G -Xptxas -O0 -lineinfo -O0 -Xcompiler -O0,${NVCC_WARNING_FLAG},-g,-ggdb3,-fPIC")
endif()

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fPIC -g -ggdb3 -O3 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -fPIC -g -ggdb3 -O3 ${CC_WARNING_FLAG}")
if(ENABLE_CUDA)
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}  -g -Xptxas -O3 -lineinfo -O3 -Xcompiler -O3,${NVCC_WARNING_FLAG},-g,-ggdb3,-fPIC")
endif()


# setting output folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# configure_file(${PROJECT_SOURCE_DIR}/HugeCTR/include/config.hpp.in ${PROJECT_SOURCE_DIR}/HugeCTR/include/config.hpp)

# building

# The container installs fmt/gflags/boost/TBB into a dependency prefix
# (dockerfiles/setup_centos7.sh: DEPS_PREFIX=${RECSTORE_DEPS_PREFIX:-$HOME/recstore_deps})
# and passes it as CMAKE_PREFIX_PATH.  Re-append it here because some build
# front ends own that variable and overwrite it (scikit-build-core FORCE-sets
# CMAKE_PREFIX_PATH from its CMake init cache), and because a configure with a
# fresh build directory otherwise cannot resolve fmt.
if(NOT RECSTORE_DEPS_PREFIX)
  if(DEFINED ENV{RECSTORE_DEPS_PREFIX})
    set(RECSTORE_DEPS_PREFIX "$ENV{RECSTORE_DEPS_PREFIX}")
  elseif(IS_DIRECTORY "$ENV{HOME}/recstore_deps/lib64/cmake")
    set(RECSTORE_DEPS_PREFIX "$ENV{HOME}/recstore_deps")
  endif()
endif()
set(RECSTORE_DEPS_PREFIX "${RECSTORE_DEPS_PREFIX}" CACHE PATH
    "Dependency prefix providing fmt/gflags/boost/TBB (see dockerfiles/setup_centos7.sh)")
if(RECSTORE_DEPS_PREFIX)
  list(APPEND CMAKE_PREFIX_PATH "${RECSTORE_DEPS_PREFIX}")
  # Same reason: the build-tree rpath used to come from a CMAKE_BUILD_RPATH
  # environment variable, so keep the prefix's libraries on the build rpath from
  # the project itself (the installed rpath stays $ORIGIN-based).
  list(APPEND CMAKE_BUILD_RPATH
       "${RECSTORE_DEPS_PREFIX}/lib"
       "${RECSTORE_DEPS_PREFIX}/lib64")
endif()

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/third_party/grpc-install")

include("${CMAKE_CURRENT_LIST_DIR}/cmake/FindGRPC.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/common.cmake")



list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/third_party/libtorch/libtorch/share/cmake/Torch")
set(Torch_DIR "${PROJECT_SOURCE_DIR}/third_party/libtorch/libtorch/share/cmake/Torch")






set(RECSTORE_FOLLY_INSTALL_DIR
    "${PROJECT_SOURCE_DIR}/third_party/folly/folly-install-fPIC/usr/local/lib/cmake/folly")
set(RECSTORE_FOLLY_BUILD_DIR "${PROJECT_SOURCE_DIR}/third_party/folly/_build")
set(RECSTORE_FOLLY_BUILD_LIBRARY "${RECSTORE_FOLLY_BUILD_DIR}/libfolly.a")
set(RECSTORE_GLOG_INSTALL_DIR
    "${PROJECT_SOURCE_DIR}/third_party/glog/glog-install-fPIC/usr/local/lib/cmake/glog")
set(folly_DIR "${RECSTORE_FOLLY_INSTALL_DIR}")
set(FOLLY_ROOT_DIR "${folly_DIR}")

set(FMT_DEBUG_POSTFIX "" CACHE STRING "Debug library postfix." FORCE)
add_subdirectory(third_party/fmt)

add_library(jemalloc_compat STATIC
  ${PROJECT_SOURCE_DIR}/src/base/jemalloc_compat.cc
)
target_link_libraries(jemalloc_compat PUBLIC jemalloc ${CMAKE_DL_LIBS})

if(EXISTS "${RECSTORE_FOLLY_INSTALL_DIR}/folly-config.cmake")
  find_package(folly REQUIRED)
  if(TARGET Folly::folly)
    set_target_properties(Folly::folly PROPERTIES MAP_IMPORTED_CONFIG_DEBUG Release)
  endif()
elseif(EXISTS "${RECSTORE_FOLLY_BUILD_LIBRARY}")
  find_package(Threads REQUIRED)
  find_package(Boost 1.51.0 REQUIRED COMPONENTS
    context
    filesystem
    program_options
    regex
    system
    thread
  )
  add_library(Folly::folly STATIC IMPORTED)
  set_target_properties(Folly::folly PROPERTIES
    IMPORTED_LOCATION "${RECSTORE_FOLLY_BUILD_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES
      "${PROJECT_SOURCE_DIR}/third_party/folly;${RECSTORE_FOLLY_BUILD_DIR};${PROJECT_SOURCE_DIR}/third_party/liburing/src/include"
    INTERFACE_LINK_LIBRARIES
      "fmt::fmt;Threads::Threads;Boost::context;Boost::filesystem;Boost::program_options;Boost::regex;Boost::system;Boost::thread;/usr/lib/x86_64-linux-gnu/libdouble-conversion.so;/usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2;glog::glog;jemalloc_compat;/usr/lib/x86_64-linux-gnu/libevent.so;/usr/lib/x86_64-linux-gnu/libz.so;/usr/lib/x86_64-linux-gnu/libssl.so;/usr/lib/x86_64-linux-gnu/libcrypto.so;/usr/lib/x86_64-linux-gnu/libbz2.so;/usr/lib/x86_64-linux-gnu/liblzma.so;/usr/lib/x86_64-linux-gnu/liblz4.so;/usr/lib/x86_64-linux-gnu/libzstd.so;/usr/lib/x86_64-linux-gnu/libsnappy.so;/usr/local/lib/libdwarf.a;/usr/lib/x86_64-linux-gnu/libiberty.a;/usr/lib/x86_64-linux-gnu/libaio.so;${PROJECT_SOURCE_DIR}/third_party/liburing/src/liburing.a;dl;/usr/lib/x86_64-linux-gnu/libunwind.so"
  )
else()
  add_library(Folly::folly INTERFACE IMPORTED)
  target_include_directories(Folly::folly INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/folly
    ${PROJECT_SOURCE_DIR}/third_party/folly/_build
  )
endif()
find_package(gflags REQUIRED)

if(EXISTS "${PROJECT_SOURCE_DIR}/third_party/folly/folly-install-fPIC/usr/local/include")
  include_directories(${PROJECT_SOURCE_DIR}/third_party/folly/folly-install-fPIC/usr/local/include)
  link_directories(${PROJECT_SOURCE_DIR}/third_party/folly/folly-install-fPIC/usr/local/lib)
endif()
include_directories(
  ${PROJECT_SOURCE_DIR}/third_party/folly
  ${PROJECT_SOURCE_DIR}/third_party/folly/_build
)

if(Boost_INCLUDE_DIR)
  include_directories(${Boost_INCLUDE_DIR})
endif()

set(glog_DIR "${RECSTORE_GLOG_INSTALL_DIR}")
if(EXISTS "${RECSTORE_GLOG_INSTALL_DIR}/glog-config.cmake")
  find_package(glog CONFIG REQUIRED)
elseif(EXISTS "${PROJECT_SOURCE_DIR}/third_party/glog/CMakeLists.txt")
  # glog registers its unit tests with CTest. Keep them in the default build:
  # otherwise `ctest` reports six "Not Run" entries for binaries that are
  # excluded from `all`.
  add_subdirectory(third_party/glog)
else()
  find_package(glog CONFIG REQUIRED)
endif()

if(TARGET glog::glog AND NOT TARGET glog)
  add_library(glog ALIAS glog::glog)
endif()
if(TARGET gflags_shared AND NOT TARGET gflags)
  add_library(gflags ALIAS gflags_shared)
elseif(TARGET gflags::gflags AND NOT TARGET gflags)
  add_library(gflags ALIAS gflags::gflags)
endif()


if(OPENMP_FOUND)
  if(ENABLE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
  endif()
  message(STATUS "add -fopenmp to compiler")
else()
  message(FATAL_ERROR "without openmp the multi-node all2all will fail")
endif()

add_subdirectory(third_party/googletest)
add_subdirectory(third_party/json)
# add_subdirectory(third_party/pybind11)
# add_subdirectory(third_party/mimalloc)

# pybind11 is not vendored, so locate its CMake package here instead of relying
# on a pre-populated cache: a fresh configure (for example after
# scikit-build-core clears the build directory) has to stand on its own.
# -DPYBIND11_CMAKE_DIR=... overrides.
#
# The pip location is image-specific, and images that install pybind11 through
# apt (dockerfiles/init_env_inside_docker.sh) keep it under the default system
# prefix instead, so a miss here is not an error: find_package() below performs
# the real search and reports anything missing itself.
#
# Note: do not call find_package(Python ...) before find_package(pybind11) here.
# That switches pybind11 into its FindPython ("NewTools") mode, which needs a
# pybind11 new enough to define python_add_library; some images ship an older
# pybind11 that only supports the classic mode.
if(NOT DEFINED PYBIND11_CMAKE_DIR)
  file(GLOB _recstore_pybind11_candidates
    "/usr/local/lib/python3.*/dist-packages/pybind11/share/cmake/pybind11"
    "/usr/local/lib/python3.*/site-packages/pybind11/share/cmake/pybind11"
    "/usr/local/python3.*/lib/python3.*/site-packages/pybind11/share/cmake/pybind11"
    "$ENV{HOME}/.local/lib/python3.*/site-packages/pybind11/share/cmake/pybind11")
  if(Python_SITELIB)
    list(APPEND _recstore_pybind11_candidates
      "${Python_SITELIB}/pybind11/share/cmake/pybind11")
  endif()
  foreach(_recstore_pybind11_candidate IN LISTS _recstore_pybind11_candidates)
    if(EXISTS "${_recstore_pybind11_candidate}/pybind11Config.cmake")
      set(PYBIND11_CMAKE_DIR "${_recstore_pybind11_candidate}")
      break()
    endif()
  endforeach()
endif()
if(EXISTS "${PYBIND11_CMAKE_DIR}/pybind11Config.cmake")
  list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
endif()
find_package(pybind11 REQUIRED)

set(CTEST_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/Testing")
enable_testing()

include_directories(
  ${PROJECT_SOURCE_DIR}/src
)
include_directories(
  ${PROJECT_SOURCE_DIR}
)
if(EXISTS "${PROJECT_SOURCE_DIR}/third_party/folly/folly-install-fPIC/usr/local/include")
  include_directories(
    ${PROJECT_SOURCE_DIR}/third_party/folly/folly-install-fPIC/usr/local/include
  )
endif()
include_directories(
  ${PROJECT_SOURCE_DIR}/third_party/folly
  ${PROJECT_SOURCE_DIR}/third_party/folly/_build
)


set(ONEAPI_ROOT /opt/intel/oneapi)
set(TBB_ROOT ${ONEAPI_ROOT}/tbb/latest/lib)
list(APPEND CMAKE_PREFIX_PATH ${TBB_ROOT}/cmake)
find_package(TBB REQUIRED)


message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
add_subdirectory(${PROJECT_SOURCE_DIR}/src)

set(YCSB_SRC_DIR ${CMAKE_SOURCE_DIR}/third_party/ycsb)
set(YCSB_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ycsb-build)
option(RECSTORE_ENABLE_HDRHISTOGRAM_PROGRAMS "Build HdrHistogram programs/tests under third_party/ycsb" OFF)
set(YCSB_REQUIRED_FILES
  "${YCSB_SRC_DIR}/CMakeLists.txt"
  "${YCSB_SRC_DIR}/core/client.h"
  "${YCSB_SRC_DIR}/core/measurements.h"
  "${YCSB_SRC_DIR}/HdrHistogram_c/CMakeLists.txt"
)
set(YCSB_SOURCE_COMPLETE ON)
foreach(YCSB_REQUIRED_FILE ${YCSB_REQUIRED_FILES})
  if(NOT EXISTS "${YCSB_REQUIRED_FILE}")
    set(YCSB_SOURCE_COMPLETE OFF)
  endif()
endforeach()
if (YCSB_SOURCE_COMPLETE)
  if(NOT RECSTORE_ENABLE_HDRHISTOGRAM_PROGRAMS)
    # Keep hdr_histogram library build, but skip third-party tests/examples registration in CTest.
    set(HDR_HISTOGRAM_BUILD_PROGRAMS OFF CACHE BOOL "Build tests and examples" FORCE)
  endif()
  add_subdirectory(${YCSB_SRC_DIR})
elseif(EXISTS "${YCSB_SRC_DIR}/CMakeLists.txt")
  message(STATUS "Skipping third_party/ycsb because its vendored sources are incomplete")
endif()
