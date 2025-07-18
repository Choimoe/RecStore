# Find the PyTorch package.
# This will now automatically find the version installed in your Python environment
# via pip, ensuring perfect compatibility.
find_package(Torch REQUIRED)

# Find the Python interpreter and development headers.
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)

# Define the C++ extension as a shared library.
add_library(recstore_torch_ops SHARED
    op_torch.cc
    ../op.cc
    ../shm_kv_store/shm_kv_store.cc
)

# Specify the include directories required for compilation.
target_include_directories(recstore_torch_ops PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

# Link the compiled library against the libraries provided by PyTorch.
target_link_libraries(recstore_torch_ops PRIVATE
    ${TORCH_LIBRARIES}
)

# Set the required C++ standard.
set_target_properties(recstore_torch_ops PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Set the output name of the library.
set_target_properties(recstore_torch_ops PROPERTIES
    OUTPUT_NAME "_recstore_ops"
)

# --- Testing ---
add_test(
    NAME pytorch_client_test
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python_client/client_test.py
            "$<TARGET_FILE:recstore_torch_ops>"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python_client
)

add_test(
    NAME dist_emb_unittest
    COMMAND ${Python_EXECUTABLE} -m unittest recstore.unittest.test_dist_emb
)

set_tests_properties(dist_emb_unittest PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/pytorch
)
