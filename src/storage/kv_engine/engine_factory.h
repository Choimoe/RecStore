#pragma once

#include "../kv_engine/engine_cceh.h"
#include "../kv_engine/engine_extendible_hash.h"
#include "../kv_engine/engine_hybridkv.h"
#include "../kv_engine/engine_map.h"
#include "../kv_engine/engine_pethash.h"
#include "storage/kv_engine/kv_engine.h"

namespace base {

inline void RegisterKVEngineFactories() {
  FactoryCreatorImpl<BaseKV, KVEngine, const BaseKVConfig&>("KVEngine");
  FactoryCreatorImpl<BaseKV,
                     KVEngineExtendibleHash,
                     const BaseKVConfig&>("KVEngineExtendibleHash");
  FactoryCreatorImpl<BaseKV, KVEngineMap, const BaseKVConfig&>("KVEngineMap");
  FactoryCreatorImpl<BaseKV, KVEngineCCEH, const BaseKVConfig&>(
      "KVEngineCCEH");
  FactoryCreatorImpl<BaseKV, KVEngineHybrid, const BaseKVConfig&>(
      "KVEngineHybrid");
  FactoryCreatorImpl<BaseKV, KVEnginePetKV, const BaseKVConfig&>(
      "KVEnginePetKV");

  FactoryCreatorImpl<Index, DramExtendibleHashIndex, const BaseKVConfig&>(
      "DRAM_EXTENDIBLE_HASH");
  FactoryCreatorImpl<Index, DramExtendibleHashIndex, const BaseKVConfig&>(
      "DRAM");
  FactoryCreatorImpl<Index, DramUnorderedMapIndex, const BaseKVConfig&>(
      "DRAM_UNORDERED_MAP");
  FactoryCreatorImpl<Index, DramPetHashIndex, const BaseKVConfig&>(
      "DRAM_PET_HASH");
  FactoryCreatorImpl<Index, SsdExtendibleHashIndex, const BaseKVConfig&>(
      "SSD");
  FactoryCreatorImpl<Index, SsdExtendibleHashIndex, const BaseKVConfig&>(
      "SSD_EXTENDIBLE_HASH");

  FactoryCreatorImpl<ValueStore, DramValueStore, const BaseKVConfig&>(
      "DRAM_VALUE_STORE");
  FactoryCreatorImpl<ValueStore, SsdValueStore, const BaseKVConfig&>(
      "SSD_VALUE_STORE");
  FactoryCreatorImpl<ValueStore, HybridValueStore, const BaseKVConfig&>(
      "TIERED_VALUE_STORE");
}

} // namespace base
