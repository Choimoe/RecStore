#pragma once

#include "storage/kv_engine/kv_engine.h"

namespace base {

inline void RegisterKVEngineFactories() {
  FactoryCreatorImpl<BaseKV, KVEngine, const BaseKVConfig&>("KVEngine");
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
