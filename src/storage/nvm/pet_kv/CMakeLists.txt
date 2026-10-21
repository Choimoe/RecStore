add_library(pet_kv pet_kv.cc)
target_link_libraries(pet_kv PUBLIC base base_memory)
target_compile_options(pet_kv PUBLIC -msse -msse2 -msse3 -msse4)
target_compile_options(pet_kv PUBLIC -Wno-error=class-memaccess)


option(USE_PIBENCH "Use dash hash" OFF)
set(PIBENCH_ROOT "" CACHE PATH "Path to the PiBench/HashEvaluation source tree")
if(USE_PIBENCH)
  if(NOT EXISTS "${PIBENCH_ROOT}/hash/common/hash_api.h")
    message(FATAL_ERROR
      "USE_PIBENCH requires -DPIBENCH_ROOT=/path/to/HashEvaluation")
  endif()
  add_library(
    PiBenchPetHash SHARED
    PiBenchWrapper.cc
    # pet_kv.cc
    # old_shm_kv.cc
  )
  target_compile_definitions(PiBenchPetHash PRIVATE USED_IN_PI_BENCH)
  target_include_directories(PiBenchPetHash PRIVATE "${PIBENCH_ROOT}/hash/common")
  target_link_libraries(PiBenchPetHash PRIVATE base base_memory)
  target_compile_options(PiBenchPetHash PRIVATE -msse -msse2 -msse3 -msse4)
  target_compile_options(PiBenchPetHash PRIVATE -Wno-error=class-memaccess)
  add_custom_command(
    TARGET PiBenchPetHash
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PIBENCH_ROOT}/bin"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:PiBenchPetHash>" "${PIBENCH_ROOT}/bin/")
endif()
