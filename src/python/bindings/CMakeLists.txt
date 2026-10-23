pybind11_add_module(_recstore_kv MODULE
  recstore_kv_pybind.cc
  recstore_kv_bindings.cc
  local_kv_binding.cc
  local_indexed_kv_binding.cc
  local_indexed_kv_core.cc
  local_indexed_kv_lookup.cc
  local_indexed_kv_checkpoint.cc
  local_indexed_kv_optimizer.cc
  local_kv_engine_register.cc
)

# The Python module only uses KVEngineComposite. Linking the complete
# kv_engine archive also pulls the SSD/external engines into the extension,
# adding a large initial-exec TLS segment and unrelated runtime dependencies.
# local_kv_engine_register.cc provides the one factory registration needed by
# the in-process binding.
include(ExternalProject)
set(_recstore_glog_noflags_build
  "${CMAKE_CURRENT_BINARY_DIR}/glog-noflags")
set(_recstore_glog_noflags_archive
  "${_recstore_glog_noflags_build}/libglog.a")
ExternalProject_Add(recstore_python_glog_noflags_build
  SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/glog"
  BINARY_DIR "${_recstore_glog_noflags_build}"
  CMAKE_ARGS
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DBUILD_SHARED_LIBS=OFF
    -DBUILD_TESTING=OFF
    -DWITH_GFLAGS=OFF
    -DWITH_GTEST=OFF
  BUILD_COMMAND
    "${CMAKE_COMMAND}" --build <BINARY_DIR> --target glog --parallel
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS "${_recstore_glog_noflags_archive}"
  EXCLUDE_FROM_ALL TRUE
)
add_library(recstore_python_glog_noflags STATIC IMPORTED GLOBAL)
set_target_properties(recstore_python_glog_noflags PROPERTIES
  IMPORTED_LOCATION "${_recstore_glog_noflags_archive}"
)
add_dependencies(recstore_python_glog_noflags
  recstore_python_glog_noflags_build)

target_link_options(_recstore_kv PRIVATE -Wl,--as-needed)
target_compile_options(_recstore_kv PRIVATE -msse4.2)
target_link_libraries(_recstore_kv PRIVATE
  -Wl,--whole-archive
  recstore_python_glog_noflags
  -Wl,--no-whole-archive
  base
  base_memory
  extendible_hash
  OpenMP::OpenMP_CXX
)

target_compile_definitions(_recstore_kv PRIVATE
  RECSTORE_VERSION="${RECSTORE_VERSION}"
  RECSTORE_KV_API_VERSION=2
  RECSTORE_KV_ABI_VERSION=1
)

set_target_properties(_recstore_kv PROPERTIES
  INSTALL_RPATH "$ORIGIN/native"
  INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE
)

install(TARGETS _recstore_kv
  LIBRARY DESTINATION recstore
  COMPONENT python
)

add_test(NAME recstore_kv_binding_metadata
  COMMAND ${CMAKE_COMMAND} -E env
    "LD_PRELOAD=$<TARGET_FILE:_recstore_kv>"
    "PYTHONPATH=$<TARGET_FILE_DIR:_recstore_kv>"
    "${PYTHON_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_recstore_kv_metadata.py"
)

add_test(NAME recstore_kv_binding_functional
  COMMAND ${CMAKE_COMMAND} -E env
    "LD_PRELOAD=$<TARGET_FILE:_recstore_kv>"
    "PYTHONPATH=$<TARGET_FILE_DIR:_recstore_kv>"
    "${PYTHON_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_recstore_kv_functional.py"
)
