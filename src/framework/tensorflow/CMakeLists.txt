# cmake_minimum_required(VERSION 3.10)
project(recstore_tf_ops)

execute_process(
    COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
    OUTPUT_VARIABLE TENSORFLOW_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
    OUTPUT_VARIABLE TENSORFLOW_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

file(GLOB TENSORFLOW_FRAMEWORK_SO ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so*)
file(GLOB TENSORFLOW_CC_SO ${TENSORFLOW_LIB_DIR}/libtensorflow_cc.so*)

set(TENSORFLOW_LIBRARIES
    ${TENSORFLOW_FRAMEWORK_SO}
    ${TENSORFLOW_CC_SO}
)

find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)

# --- Build shared library ---
add_library(recstore_tf_ops SHARED
    op_tensorflow.cc
    ../op.cc
)

if(USE_FAKE_KVCLIENT)
    target_compile_definitions(recstore_tf_ops PRIVATE USE_FAKE_KVCLIENT)
endif()

include_directories(${PROJECT_SOURCE_DIR}/src/framework)

# --- Include directories ---
target_include_directories(recstore_tf_ops PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${TENSORFLOW_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
)

# --- Link libraries ---
target_link_libraries(recstore_tf_ops PRIVATE
    ${TENSORFLOW_LIBRARIES}
)

# --- Set C++ standard ---
set_target_properties(recstore_tf_ops PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "_recstore_ops"
)

# --- Tests ---
add_test(
    NAME tf_client_test
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/python_client/client_test.py
            "$<TARGET_FILE:recstore_tf_ops>"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python_client
)

# add_test(
#     NAME dist_emb_unittest
#     COMMAND ${Python3_EXECUTABLE} -m unittest recstore.unittest.test_dist_emb
# )

# set_tests_properties(dist_emb_unittest PROPERTIES
#     WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/tensorflow
# )
