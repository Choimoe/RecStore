# Batch Storage Benchmark Tools

This directory contains reusable helpers for running dense one-dimensional
`backend_benchmark` scans. The scripts are intentionally filesystem based:
each parameter point gets its own directory with `summary.csv`,
`aggregate.csv`, `manifest.jsonl`, per-run logs, and optional backend data.

## Scripts

- `storage_runner.py`: runs one parameter point for one or more backends and
  repeats. It records timeout as `exit_code=124` and aggregates only successful
  `phase=run` rows.
- `run_dense_scans.py`: schedules dense scans for thread count, value size,
  batch size, Zipf alpha, and read ratio. It can resume by skipping points that
  already have enough successful run rows.
- `summarize_status.py`: reports completion, success counts, failure counts,
  and failure exit codes for a scan directory.
- `plot_dense_scans.py`: creates aggregate CSV files and PDF line charts with
  error bars from the scan outputs.

## Example

```bash
EXP=/tmp/recstore-dense-scan

python3 tools/benchmarks/batch_test/run_dense_scans.py \
  --output-root "$EXP" \
  --backends fasterkv dram_pet_dram dram_eh_dram hps_rocksdb \
  --repeat 6 \
  --record-count 1000000 \
  --runtime-seconds 5 \
  --timeout-seconds 180

python3 tools/benchmarks/batch_test/summarize_status.py "$EXP" --show-complete

python3 tools/benchmarks/batch_test/plot_dense_scans.py "$EXP" \
  --aggregate-dir "$EXP/aggregate" \
  --figure-dir "$EXP/figures"
```

To run only a subset:

```bash
python3 tools/benchmarks/batch_test/run_dense_scans.py \
  --output-root "$EXP" \
  --scans thread value \
  --thread-values "1 2 4 8 16 32" \
  --value-sizes "64 128 256 512 1024"
```

## Output Contract

For each point, `raw/<label>/summary.csv` keeps all parsed rows, including
failed and timeout samples. `raw/<label>/aggregate.csv` contains only successful
steady-state `phase=run` rows. Failed samples must not be treated as zero
throughput; use `summarize_status.py` to report them separately.

The plotting script skips directories containing `_aborted_` in the name, which
lets operators preserve manually interrupted runs without polluting official
curves.
