# Copyright 2023 Chengye YU <yuchengye2013 AT outlook.com>.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.10)
project(ycsb-cpp LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Release)
set(CompilerFlags
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
        )

if(MSVC)
    foreach(CompilerFlag ${CompilerFlags})
        string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
    endforeach()
endif()

option(BIND_ROCKSDB "build with rocksdb" OFF)
option(BIND_LMDB "build with lmdb" OFF)
option(BIND_LEVELDB "build with leveldb" OFF)
option(BIND_SQLITE "build with sqlite" OFF)
option(BIND_WIREDTIGER "build with wiredtiger" OFF)
option(BIND_RECSTORE "build with RecStore kvdb/cceh bindings" ON)

option(WITH_ZLIB "linking YCSB with zlib; needed by HdrHISTOGRAM, DO NOT TURN OFF" ON)
option(WITH_LZ4 "linking YCSB with lz4" OFF)
option(WITH_SNAPPY "linking YCSB with snappy" OFF)
option(WITH_ZSTD "linking YCSB with zstd" OFF)
option(WITH_BZ2 "linking YCSB with bzip2" OFF)

file(GLOB_RECURSE YCSB_CORE_SRC "core/*.cc")

include(FindThreads)

if (NOT TARGET fmt::fmt)
    set(FMT_DEBUG_POSTFIX "" CACHE STRING "Debug library postfix." FORCE)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../fmt" "${CMAKE_BINARY_DIR}/fmt")
endif()

add_executable(ycsb_basic ${YCSB_CORE_SRC})
target_include_directories(ycsb_basic PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(ycsb_basic PRIVATE Threads::Threads)
set_target_properties(ycsb_basic PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(ycsb ${YCSB_CORE_SRC})
target_include_directories(ycsb PRIVATE ${PROJECT_SOURCE_DIR})

target_link_libraries(ycsb PRIVATE fmt::fmt)

if (BIND_ROCKSDB)
    message(STATUS "BIND_ROCKSDB - ON")
    set(WITH_ZLIB ON)
    set(WITH_BZ2 ON)
    file(GLOB_RECURSE YCSB_ROCKSDB_SRC "rocksdb/*.cc")
    target_sources(ycsb PRIVATE ${YCSB_ROCKSDB_SRC})

    find_package(RocksDB CONFIG)
    if(RocksDB_FOUND)
        if(MSVC)
            target_link_libraries(ycsb PRIVATE RocksDB::rocksdb rpcrt4.lib)
        else()
            target_link_libraries(ycsb PRIVATE RocksDB::rocksdb)
        endif()
    else()
        message(STATUS "Try to find rocksdb library using find_library")
        find_library(ROCKSDB_LIB rocksdb REQUIRED)
        find_path(ROCKSDB_INCLUDE_DIR "rocksdb/db.h" REQUIRED)
        target_link_libraries(ycsb PRIVATE ${ROCKSDB_LIB})
        target_include_directories(ycsb PRIVATE ${ROCKSDB_INCLUDE_DIR})
    endif()
    message(STATUS "Found RocksDB library")
    if(NOT MSVC)
        target_link_libraries(ycsb PRIVATE dl)
    endif()

    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_QUIET ON)
    set(_YCSB_PREVIOUS_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
    if(ROCKSDB_INCLUDE_DIR)
        set(CMAKE_REQUIRED_INCLUDES "${ROCKSDB_INCLUDE_DIR}")
    endif()
    check_cxx_source_compiles("
        #include <memory>
        #include <string>
        #include <rocksdb/env.h>
        #include <rocksdb/status.h>
        #include <rocksdb/utilities/options_util.h>
        int main() {
          rocksdb::Env* env = nullptr;
          std::shared_ptr<rocksdb::Env> guard;
          rocksdb::Status s = rocksdb::Env::CreateFromUri(
              rocksdb::ConfigOptions(), std::string(), std::string(), &env, &guard);
          return s.ok() ? 0 : 0;
        }
    " YCSB_HAVE_ROCKSDB_ENV_CREATE_FROM_URI)
    set(CMAKE_REQUIRED_INCLUDES "${_YCSB_PREVIOUS_REQUIRED_INCLUDES}")
    unset(_YCSB_PREVIOUS_REQUIRED_INCLUDES)
    if(YCSB_HAVE_ROCKSDB_ENV_CREATE_FROM_URI)
        target_compile_definitions(ycsb PRIVATE YCSB_HAVE_ROCKSDB_ENV_CREATE_FROM_URI=1)
    endif()
else()
    message(STATUS "BIND_ROCKSDB - OFF")
endif()

if (BIND_LMDB)
    message(STATUS "BIND_LMDB - ON")
    file(GLOB_RECURSE YCSB_LMDB_SRC "lmdb/*.cc")
    target_sources(ycsb PRIVATE ${YCSB_LMDB_SRC})

    find_package(lmdb CONFIG)
    if(lmdb_FOUND)
        target_link_libraries(ycsb PRIVATE lmdb)
    else()
        message(STATUS "Try to find lmdb library using find_library")
        find_library(LMDB_LIB lmdb REQUIRED)
        find_path(LMDB_INCLUDE_DIR "lmdb.h" REQUIRED)
        target_link_libraries(ycsb PRIVATE ${LMDB_LIB})
        target_include_directories(ycsb PRIVATE ${LMDB_INCLUDE_DIR})
    endif()
    message(STATUS "Found LMDB library")
else()
    message(STATUS "BIND_LMDB - OFF")
endif()

if (BIND_LEVELDB)
    message(STATUS "BIND_LEVELDB - ON")
    file(GLOB_RECURSE YCSB_LEVELDB_SRC "leveldb/*.cc")
    target_sources(ycsb PRIVATE ${YCSB_LEVELDB_SRC})

    find_package(leveldb CONFIG)
    if(leveldb_FOUND)
        target_link_libraries(ycsb PRIVATE leveldb::leveldb)
    else()
        message(STATUS "Try to find leveldb library using find_library")
        find_library(LEVELDB_LIB leveldb REQUIRED)
        find_path(LEVELDB_INCLUDE_DIR "leveldb/db.h" REQUIRED)
        target_link_libraries(ycsb PRIVATE ${LEVELDB_LIB})
        target_include_directories(ycsb PRIVATE ${LEVELDB_INCLUDE_DIR})
    endif()
    message(STATUS "Found leveldb library")
else()
    message(STATUS "BIND_LEVELDB - OFF")
endif ()

if (BIND_SQLITE)
    message(STATUS "BIND_SQLITE - ON")
    file(GLOB_RECURSE YCSB_SQLITE_SRC "sqlite/*.cc")
    target_sources(ycsb PRIVATE ${YCSB_SQLITE_SRC})
    find_package(SQLite3)
    if(SQLite3_FOUND)
        target_link_libraries(ycsb PRIVATE SQLite::SQLite3)
    else()
        message(STATUS "Try to find sqlite3 library using find_library")
        find_library(SQLITE3_LIB sqlite3 REQUIRED)
        find_path(SQLITE3_INCLUDE_DIR "sqlite3.h" REQUIRED)
        target_link_libraries(ycsb PRIVATE ${SQLITE3_LIB})
        target_include_directories(ycsb PRIVATE ${SQLITE3_INCLUDE_DIR})
    endif()
    message(STATUS "Found sqlite3 library")
else()
    message(STATUS "BIND_SQLITE - OFF")
endif()

if (BIND_WIREDTIGER)
    message(STATUS "BIND_WIREDTIGER - ON")
    file(GLOB_RECURSE YCSB_WT_SRC "wiredtiger/*.cc")
    target_sources(ycsb PRIVATE ${YCSB_WT_SRC})
    find_library(WIREDTIGER_LIB wiredtiger REQUIRED)
    find_path(WIREDTIGER_INCLUDE_DIR "wiredtiger.h" REQUIRED)
    target_link_libraries(ycsb PRIVATE ${WIREDTIGER_LIB})
    target_include_directories(ycsb PRIVATE ${WIREDTIGER_INCLUDE_DIR})
else()
    message(STATUS "BIND_WIREDTIGER - OFF")
endif()

if(WITH_ZLIB)
    message(STATUS "WITH_ZLIB - ON")
    find_package(ZLIB)
    if(ZLIB_FOUND)
        target_link_libraries(ycsb PRIVATE ZLIB::ZLIB)
    else()
        message(STATUS "Try to find zlib library using find_library")
        find_library(ZLIB_LIB zlib)
        if(ZLIB_LIB-NOTFOUND)
            find_library(ZLIB_LIB z REQUIRED)
        endif()
        target_link_libraries(ycsb PRIVATE ${ZLIB_LIB})
    endif()
    message(STATUS "Found ZLIB library")
else()
    message(STATUS "WITH_ZLIB - OFF")
endif()

if(WITH_SNAPPY)
    message(STATUS "WITH_SNAPPY - ON")
    find_package(Snappy CONFIG)
    if(Snappy_FOUND)
        target_link_libraries(ycsb PRIVATE Snappy::snappy)
    else()
        message(STATUS "Try to find Snappy library using find_library")
        find_library(SNAPPY_LIB snappy REQUIRED)
        target_link_libraries(ycsb PRIVATE ${SNAPPY_LIB})
    endif()
    message(STATUS "Found Snappy library")
else()
    message(STATUS "WITH_SNAPPY - OFF")
endif()

if(WITH_LZ4)
    message(STATUS "WITH_LZ4 - ON")
    find_package(lz4 CONFIG)
    if(lz4_FOUND)
        target_link_libraries(ycsb PRIVATE lz4::lz4)
    else()
        message(STATUS "Try to find LZ4 library using find_library")
        find_library(LZ4_LIB lz4 REQUIRED)
        target_link_libraries(ycsb PRIVATE ${LZ4_LIB})
    endif()
    message(STATUS "Found LZ4 library")
else()
    message(STATUS "WITH_LZ4 - OFF")
endif()

if(WITH_ZSTD)
    message(STATUS "WITH_ZSTD - ON")
    find_package(zstd CONFIG)
    if(zstd_FOUND)
        target_link_libraries(ycsb PRIVATE $<IF:$<TARGET_EXISTS:zstd::libzstd_shared>,zstd::libzstd_shared,zstd::libzstd_static>)
    else()
        message(STATUS "Try to find zstd library using find_library")
        find_library(ZSTD_LIB zstd REQUIRED)
        target_link_libraries(ycsb PRIVATE ${ZSTD_LIB})
    endif()
    message(STATUS "Found zstd library")
else()
    message(STATUS "WITH_ZSTD - OFF")
endif()

if(WITH_BZ2)
    message(STATUS "WITH_BZ2 - ON")
    find_package(BZip2)
    if(BZip2_FOUND)
        target_link_libraries(ycsb PRIVATE BZip2::BZip2)
    else()
        message(STATUS "Try to find BZip2 library using find_library")
        find_library(BZ2_LIB bz2 REQUIRED)
        target_link_libraries(ycsb PRIVATE ${BZ2_LIB})
    endif()
    message(STATUS "Found BZip2 library")
else()
    message(STATUS "WITH_BZ2 - OFF")
endif()


add_subdirectory(HdrHistogram_c)
include_directories(HdrHistogram_c/include)
target_compile_definitions(ycsb PRIVATE HDRMEASUREMENT)
add_dependencies(ycsb hdr_histogram_static)
target_link_libraries(ycsb PRIVATE hdr_histogram_static)

#target_compile_options(ycsb PRIVATE -fsanitize=address -fno-omit-frame-pointer -g -O1)
#target_link_options(ycsb PRIVATE -fsanitize=address)
find_package(TBB REQUIRED)
target_link_libraries(ycsb PRIVATE TBB::tbb)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (BIND_RECSTORE)
  file(GLOB YCSB_DB_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/db/*.cc")
  target_sources(ycsb PRIVATE ${YCSB_DB_SRCS})

  target_include_directories(ycsb PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
  )

  target_link_libraries(ycsb PRIVATE
    kv_engine
    base
    ps_common
    pthread
    OpenMP::OpenMP_CXX
    jemalloc
  )
else()
  message(STATUS "BIND_RECSTORE - OFF")
endif()

# ④ gperftools 存在就连；否则用本地 stub
find_library(GPERF_PROFILER_LIB profiler)
if (GPERF_PROFILER_LIB)
  target_compile_definitions(ycsb PRIVATE YCSB_USE_GPERFTOOLS)
  target_link_libraries(ycsb PRIVATE ${GPERF_PROFILER_LIB})
endif()

# ⑤ 可选：统一输出目录
set_target_properties(ycsb PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ⑥（可选）安静掉 posix_memalign 未检查返回值等告警
target_compile_options(ycsb PRIVATE
  -Wno-maybe-uninitialized -Wno-unused-result
)
