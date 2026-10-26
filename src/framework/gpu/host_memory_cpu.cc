#include "framework/gpu/host_memory.h"
namespace recstore::framework {
bool EnsurePinnedLocalShmPayload(const void*, std::size_t) { return false; }
void RetainUntilStreamComplete(void*, std::function<void()> release) {
  release();
}
} // namespace recstore::framework
