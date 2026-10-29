#include "counter.h"

#include <map>
#include <time.h>

namespace base {

static std::map<std::string, Counter*> counters;
static base::Lock counters_lock;

void Counter::Inc(int64 count) {
  // The shortest statistics window is one second; avoid reading a costly
  // high-resolution clock on every increment (for example, with HPET).
  timespec timestamp;
  int64 now;
  if (clock_gettime(CLOCK_REALTIME_COARSE, &timestamp) == 0) {
    now = timestamp.tv_sec * 1000000L + timestamp.tv_nsec / 1000;
  } else {
    now = base::GetTimestamp();
  }
  second_count_.Update(now, count_, 1000000L);
  minute_count_.Update(now, count_, 1000000L * 60);
  hour_count_.Update(now, count_, 1000000L * 3600);
  day_count_.Update(now, count_, 1000000L * 3600L * 24L);
  count_ += count;
}

std::string Counter::Display() const {
  return base::StringPrintf(
      "[%s] qps: %ld, minute: %ld, hour: %ld, day: %ld, total: %ld",
      name_.c_str(),
      second_count_.GetCount(1000000L),
      minute_count_.GetCount(1000000L * 60),
      hour_count_.GetCount(1000000L * 3600),
      day_count_.GetCount(1000000L * 3600L * 24L),
      count_);
}

} // namespace base
