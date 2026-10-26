#pragma once
#include <cstddef>
#include <functional>
namespace recstore::framework {
bool EnsurePinnedLocalShmPayload(const void* ptr, std::size_t bytes);
// The release callback runs on a CUDA host callback and must not call CUDA.
void RetainUntilStreamComplete(void* stream, std::function<void()> release);
} // namespace recstore::framework
