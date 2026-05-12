#include "storage/kv_engine/engine_composite.h"

#include "gflags/gflags.h"

DEFINE_int32(prefetch_method,
             0,
             "PetKV BatchGet prefetch method: 0=single Get loop, 1=BatchGet");
