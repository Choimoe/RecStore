# PS pipeline tuning

The base `PSPipeline` keeps embeddings authoritative in the PS and waits for the global SGD commit before consuming dependent rows. GPU embedding values belong to the current batch. The controls below apply to the eager two-rank dependency-split path.

## Trainer controls

Set these variables explicitly in every trainer process, including remote workers:

```bash
RECSTORE_PS_METADATA_NCCL=1
RECSTORE_PS_SEGMENT_REDUCE=1
RECSTORE_PS_GPU_TICKET=1
RECSTORE_FULL_LOOP_TIMING=1
```

Append `--read-mode ps_pipeline --ps-pipeline-dependency-split --ps-pipeline-early-submit` to the corresponding `rs_demo` training command. Keep CUDA graphs disabled for this eager configuration. A launcher must propagate the variables to remote processes; exporting them in the coordinator's shell alone is insufficient.

NCCL metadata uses a separate process group with bounded, fixed-capacity transfers. The pipeline requires CUDA, an NCCL WORLD group, and an explicit `max_batch_size`; `rs_demo` supplies its configured batch size. Both ranks must agree on that bound, feature count, and step count. Metadata submissions stay on the training thread, with transfer buffers retained until completion. Asynchronous dense-preparation modes are rejected.

GPU tickets construct integer IDs and stable segment plans on CUDA, then return ready CPU tickets for PS requests. Segment reduction keeps sample-major repeated-key addition order. These two switches default off; unsupported CPU, owner, dimension-one, or missing-operation cases retain the existing reduction path. The retained plans contain integer metadata and do not add an embedding cache. Measure the combination separately from segment reduction alone, since CPU plan construction can offset its GPU savings.

`RECSTORE_SPARSE_COMMIT_RESERVE` controls the server's gradient aggregation preallocation and defaults on. Set it to `0` in each PS process for its allocation ablation.

`RECSTORE_SPARSE_COMMIT_SORTED_PAIR=1` enables a two-way aggregation path when both ranks submit strictly increasing unique keys. It preserves first-seen output order and the original FP32 rank/chunk addition order. The whole input is checked before aggregation; unsorted or duplicate keys and other world sizes use the hash-based path. The switch defaults off and the existing profile reports actual `sorted_pair` hits.

`RECSTORE_SPARSE_COMMIT_POW2_SCALE=1` enables exact binary reciprocal scaling for power-of-two world divisors in each PS process. Rank/chunk addition order and the separate SGD pass stay unchanged; other divisors retain division. This defaults off. Validate numerical equality and full-loop throughput independently, since faster normalization alone does not establish an end-to-end gain.

Set `RECSTORE_SGD_ROW_PREFETCH=1` in each PS process to prefetch upcoming direct DRAM rows during sparse SGD. The update loop looks eight rows ahead and hints up to 512 bytes per row, while preserving row order and floating-point operations. This option defaults off and skips missing rows; its benefit depends on the working set, so compare full-loop throughput with the same option disabled.

`RECSTORE_DRAM_SIZE_PREFETCH=8` or `16` prefetches legacy concurrent-slab allocation-size headers during direct row lookup. The default is `0`; other allocators retain their existing lookup path. This changes metadata access, not payload size or SGD arithmetic. Compare it independently from row-data prefetch, including warm and tail batches.

## Size the PS index independently of the model

`cache_ps.base_kv_config.capacity` sizes the index. Choose it from the complete key population, including optimizer namespaces and expected growth, and verify the most populated shard. Keep model cardinalities, initialized rows, and the value allocator unchanged when testing index capacity. A cap on each feature's cardinality is different from the sum of all feature cardinalities.

For example, the SGD-only workload with 1,721,198 model rows can use this runtime-configuration fragment on both PS shards:

```json
{
  "cache_ps": {
    "base_kv_config": {
      "capacity": 2000000
    }
  }
}
```

Here 2,000,000 exceeds even the full unsharded row population. With the current 320-byte PET hash chunks, reducing the requested capacity from 10,400,000 to 2,000,000 reduces the index allocation from 320 MiB to 80 MiB, plus the 576-byte header. This is CPU index allocation, separate from the DRAM value pool and GPU memory. The smaller allocation changes bucket placement and collision behavior, so its throughput still requires measurement on the actual key distribution.

For comparisons, enable full-loop timing on both RecStore and TorchRec and compute throughput as total actual samples divided by the maximum rank elapsed time. Include initial batch consumption, every step, and the final commit/barrier/device drain. Run state validation separately from profiler-free throughput repeats, and retain every repeat in the report.
