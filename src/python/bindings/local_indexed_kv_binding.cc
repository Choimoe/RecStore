#include "local_indexed_kv.h"

namespace recstore::python {

void BindLocalIndexedKV(py::module_& m) {
  py::class_<LocalIndexedKV>(m, "LocalIndexedKV")
      .def(py::init<int64_t,
                    int64_t,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    int64_t,
                    int,
                    int>(),
           py::arg("capacity"),
           py::arg("embedding_dim"),
           py::arg("index_type"),
           py::arg("value_path"),
           py::arg("allocator")          = "CONCURRENT_SLAB_MEMORY_POOL",
           py::arg("allocator_capacity") = 0,
           py::arg("num_threads")        = 1,
           py::arg("index_prefetch")     = 1)
      .def("lookup_indexed",
           &LocalIndexedKV::LookupIndexed,
           py::arg("ids"),
           py::arg("allocate_missing") = true)
      .def("lookup_indexed_profiled",
           &LocalIndexedKV::LookupIndexedProfiled,
           py::arg("ids"),
           py::arg("allocate_missing") = true)
      .def("lookup_indexed_into_raw",
           &LocalIndexedKV::LookupIndexedIntoRaw,
           py::arg("ids_address"),
           py::arg("num_rows"),
           py::arg("indices_address"),
           py::arg("values_address"),
           py::arg("found_address"),
           py::arg("allocate_missing") = true,
           py::arg("release_gil")      = true,
           py::arg("collect_profile")  = false)
      .def("lookup_indexed_fused_into_raw",
           &LocalIndexedKV::LookupIndexedFusedIntoRaw,
           py::arg("ids_address"),
           py::arg("num_rows"),
           py::arg("indices_address"),
           py::arg("values_address"),
           py::arg("found_address"),
           py::arg("allocate_missing") = true,
           py::arg("release_gil")      = true,
           py::arg("collect_profile")  = false)
      .def("lookup_indexed_legacy_into_raw",
           &LocalIndexedKV::LookupIndexedLegacyIntoRaw,
           py::arg("ids_address"),
           py::arg("num_rows"),
           py::arg("indices_address"),
           py::arg("values_address"),
           py::arg("found_address"),
           py::arg("allocate_missing") = true,
           py::arg("release_gil")      = true,
           py::arg("collect_profile")  = false)
      .def("lookup_values_into_raw",
           &LocalIndexedKV::LookupValuesIntoRaw,
           py::arg("ids_address"),
           py::arg("num_rows"),
           py::arg("values_address"),
           py::arg("release_gil") = true)
      .def("lookup_values_into_raw_profiled",
           &LocalIndexedKV::LookupValuesIntoRawProfiled,
           py::arg("ids_address"),
           py::arg("num_rows"),
           py::arg("values_address"),
           py::arg("release_gil") = true)
      .def("lookup_values_with_misses_into_raw",
           &LocalIndexedKV::LookupValuesWithMissesIntoRaw,
           py::arg("ids_address"),
           py::arg("num_rows"),
           py::arg("values_address"),
           py::arg("missing_positions_address"),
           py::arg("release_gil") = true)
      .def("lookup_values_with_misses_into_raw_profiled",
           &LocalIndexedKV::LookupValuesWithMissesIntoRawProfiled,
           py::arg("ids_address"),
           py::arg("num_rows"),
           py::arg("values_address"),
           py::arg("missing_positions_address"),
           py::arg("release_gil") = true)
      .def("size", &LocalIndexedKV::Size)
      .def("snapshot_indexed", &LocalIndexedKV::SnapshotIndexed)
      .def("delete_indexed", &LocalIndexedKV::DeleteIndexed)
      .def("clear_indexed", &LocalIndexedKV::ClearIndexed)
      .def("save_checkpoint",
           &LocalIndexedKV::SaveCheckpoint,
           py::arg("path"),
           py::arg("metadata"))
      .def("load_checkpoint",
           &LocalIndexedKV::LoadCheckpoint,
           py::arg("path"),
           py::arg("expected_metadata"))
      .def("put_indexed", &LocalIndexedKV::PutIndexed)
      .def("enable_adam_slots", &LocalIndexedKV::EnableAdamSlots)
      .def("bulk_insert_indexed",
           &LocalIndexedKV::BulkInsertIndexed,
           py::arg("ids"),
           py::arg("indices"),
           py::arg("values"),
           py::arg("release_gil") = true)
      .def("sgd_update_indexed",
           &LocalIndexedKV::SgdUpdateIndexed,
           py::arg("ids"),
           py::arg("grads"),
           py::arg("lr"),
           py::arg("release_gil") = true)
      .def("adamw_tf_update_indexed",
           &LocalIndexedKV::AdamWTFUpdateIndexed,
           py::arg("ids"),
           py::arg("grads"),
           py::arg("lr"),
           py::arg("beta1"),
           py::arg("beta2"),
           py::arg("eps"),
           py::arg("weight_decay"),
           py::arg("release_gil") = true)
      .def("adamw_tf_update_indexed_two_batches",
           &LocalIndexedKV::AdamWTFUpdateIndexedTwoBatches,
           py::arg("ids_a"),
           py::arg("grads_a"),
           py::arg("ids_b"),
           py::arg("grads_b"),
           py::arg("lr"),
           py::arg("beta1"),
           py::arg("beta2"),
           py::arg("eps"),
           py::arg("weight_decay"),
           py::arg("release_gil") = true)
      .def("optimizer_state_stats", &LocalIndexedKV::OptimizerStateStats)
      .def("set_adam_execution",
           &LocalIndexedKV::SetAdamExecution,
           py::arg("vectorized") = false,
           py::arg("prefetch_distance") = 0)
      .def("set_min_rows_per_worker",
           &LocalIndexedKV::SetMinRowsPerWorker,
           py::arg("rows"))
      .def("worker_policy", &LocalIndexedKV::WorkerPolicy)
      .def("storage_stats", &LocalIndexedKV::StorageStats);
}

} // namespace recstore::python
