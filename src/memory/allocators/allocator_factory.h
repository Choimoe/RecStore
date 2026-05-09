#pragma once

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#include "base/factory.h"
#include "base/json.h"
#include "memory/allocators/persist_loop_slab_allocator.h"
#include "memory/allocators/r2_slab_allocator.h"
#include "memory/malloc.h"

namespace base {
namespace allocators {

inline std::string Upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return s;
}

inline std::string ImplFromAllocatorType(const std::string& allocator_type) {
  const std::string t = Upper(allocator_type);
  if (t == "PERSIST_LOOP_SLAB" || t == "PERSIST_LOOP_SHM_MALLOC" ||
      t == "PERSISTLOOPSHMMALLOC")
    return "PERSIST_LOOP_SLAB";
  if (t == "R2_SLAB" || t == "R2_SHM_MALLOC" || t == "R2SHMMALLOC")
    return "R2_SLAB";
  throw std::invalid_argument("unknown allocator_type: " + allocator_type);
}

inline std::string AllocatorTypeFromImpl(const std::string& impl_name) {
  const std::string t = Upper(impl_name);
  if (t == "PERSISTLOOPSHMMALLOC" || t == "PERSIST_LOOP_SLAB")
    return "PERSIST_LOOP_SLAB";
  if (t == "R2SHMMALLOC" || t == "R2_SLAB")
    return "R2_SLAB";
  throw std::invalid_argument("unknown allocator impl: " + impl_name);
}

inline std::string
LegacyImplFromAllocatorType(const std::string& allocator_type) {
  const std::string t = Upper(allocator_type);
  if (t == "PERSIST_LOOP_SLAB")
    return "PersistLoopShmMalloc";
  if (t == "R2_SLAB")
    return "R2ShmMalloc";
  throw std::invalid_argument("unknown allocator_type: " + allocator_type);
}

inline void NormalizeAllocatorConfig(
    json& j,
    const std::string& impl_key = "value_memory_management",
    const std::string& type_key = "allocator_type") {
  const bool has_impl = j.contains(impl_key);
  const bool has_type = j.contains(type_key);

  if (has_impl && has_type) {
    const std::string type_from_impl =
        AllocatorTypeFromImpl(j.at(impl_key).get<std::string>());
    const std::string type_from_type =
        AllocatorTypeFromImpl(j.at(type_key).get<std::string>());
    if (type_from_impl != type_from_type) {
      throw std::invalid_argument(
          "allocator_type conflicts with value_memory_management");
    }
    j[type_key] = type_from_type;
    j[impl_key] = LegacyImplFromAllocatorType(type_from_type);
    return;
  }

  if (has_type) {
    const std::string type =
        AllocatorTypeFromImpl(j.at(type_key).get<std::string>());
    j[type_key] = type;
    j[impl_key] = LegacyImplFromAllocatorType(type);
    return;
  }

  if (has_impl) {
    const std::string type =
        AllocatorTypeFromImpl(j.at(impl_key).get<std::string>());
    j[type_key] = type;
    j[impl_key] = LegacyImplFromAllocatorType(type);
    return;
  }

  j[type_key] = "PERSIST_LOOP_SLAB";
  j[impl_key] = "PersistLoopShmMalloc";
}

inline std::string
ResolveAllocatorImpl(const json& j,
                     const std::string& impl_key = "value_memory_management",
                     const std::string& type_key = "allocator_type") {
  json copy = j;
  NormalizeAllocatorConfig(copy, impl_key, type_key);
  return ImplFromAllocatorType(copy.at(type_key).get<std::string>());
}

inline std::unique_ptr<base::MallocApi> CreateAllocator(
    const json& j,
    const std::string& filename,
    int64 memory_size,
    const std::string& medium,
    const std::string& impl_key = "value_memory_management",
    const std::string& type_key = "allocator_type") {
  const std::string impl = ResolveAllocatorImpl(j, impl_key, type_key);
  if (impl == "PERSIST_LOOP_SLAB") {
    return std::unique_ptr<base::MallocApi>(
        new base::PersistLoopShmMalloc(filename, memory_size, medium));
  }
  if (impl == "R2_SLAB") {
    return std::unique_ptr<base::MallocApi>(
        new base::R2alloc(filename, memory_size, medium));
  }
  throw std::invalid_argument("unknown allocator impl: " + impl);
}

} // namespace allocators
} // namespace base
