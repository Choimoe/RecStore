import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ps_dram_transport_benchmark import (  # noqa: E402
    TRANSPORT_SPECS,
    build_benchmark_cmd,
    build_runtime_config,
    collect_case_rows,
    collect_ps_result_rows,
    collect_summary_rows,
    is_port_open,
    parse_csv_list,
    print_summary_table,
    write_csv,
)


class TestRunPSDramTransportBenchmark(unittest.TestCase):
    def test_build_runtime_config_uses_dram_index_and_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_runtime_config(
                transport="BRPC",
                index_type="DRAM_PET_HASH",
                runtime_dir=Path(tmpdir),
                num_shards=2,
                base_port=25000,
                capacity=4096,
                value_size=128,
                max_keys_per_request=256,
                num_threads=8,
                dram_allocator="PERSIST_LOOP_SLAB",
                local_shm_region="unused",
                local_shm_slot_count=64,
                local_shm_ready_queue_count=1,
                local_shm_ready_queue_burst_limit=8,
                local_shm_slot_buffer_bytes=4096,
                local_shm_client_timeout_ms=1000,
                dram_capacity_multiplier=2.0,
            )
        cache_ps = config["cache_ps"]
        self.assertEqual(cache_ps["ps_type"], "BRPC")
        self.assertEqual(cache_ps["num_shards"], 2)
        self.assertEqual(
            cache_ps["base_kv_config"]["index"]["type"], "DRAM_PET_HASH"
        )
        self.assertEqual(
            cache_ps["base_kv_config"]["value"]["type"], "DRAM_VALUE_STORE"
        )
        self.assertEqual(config["client"]["port"], 25000)
        self.assertEqual(config["distributed_client"]["servers"][1]["shard"], 1)
        self.assertNotIn("local_shm", config)

    def test_rdma_config_uses_rdma_ps_type_and_dev_shm_value_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_runtime_config(
                transport="rdma",
                index_type="DRAM_EXTENDIBLE_HASH",
                runtime_dir=Path(tmpdir),
                num_shards=2,
                base_port=25000,
                capacity=1024,
                value_size=512,
                max_keys_per_request=128,
                num_threads=4,
                dram_allocator="PERSIST_LOOP_SLAB",
                local_shm_region="unused",
                local_shm_slot_count=64,
                local_shm_ready_queue_count=1,
                local_shm_ready_queue_burst_limit=8,
                local_shm_slot_buffer_bytes=8192,
                local_shm_client_timeout_ms=1000,
                dram_capacity_multiplier=2.0,
            )
        self.assertEqual(config["cache_ps"]["ps_type"], "RDMA")
        self.assertEqual(config["distributed_client"]["num_shards"], 2)
        self.assertEqual(config["distributed_client"]["servers"][1]["port"], 25001)
        value_path = config["cache_ps"]["base_kv_config"]["value"]["path"]
        self.assertTrue(value_path.startswith("/dev/shm/recstore_ps_dram_bench/"))
        self.assertIn("/rdma_dram_extendible_hash/value", value_path)

    def test_build_runtime_config_keeps_benchmark_base_port_for_single_shard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_runtime_config(
                transport="BRPC",
                index_type="DRAM_PET_HASH",
                runtime_dir=Path(tmpdir),
                num_shards=1,
                base_port=25000,
                capacity=4096,
                value_size=128,
                max_keys_per_request=256,
                num_threads=8,
                dram_allocator="PERSIST_LOOP_SLAB",
                local_shm_region="unused",
                local_shm_slot_count=64,
                local_shm_ready_queue_count=1,
                local_shm_ready_queue_burst_limit=8,
                local_shm_slot_buffer_bytes=4096,
                local_shm_client_timeout_ms=1000,
                dram_capacity_multiplier=2.0,
            )
        self.assertEqual(config["client"]["port"], 25000)
        self.assertEqual(config["cache_ps"]["servers"][0]["port"], 25000)

    def test_rdma_transport_spec_uses_petps_server(self):
        self.assertEqual(TRANSPORT_SPECS["RDMA"].server_binary, "petps_server")
        self.assertEqual(TRANSPORT_SPECS["RDMA"].base_port, 25000)

    def test_run_one_case_rdma_uses_petps_cluster_runner(self):
        from run_ps_dram_transport_benchmark import run_one_case

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config = build_runtime_config(
                transport="RDMA",
                index_type="DRAM_EXTENDIBLE_HASH",
                runtime_dir=tmp_path,
                num_shards=2,
                base_port=25000,
                capacity=1024,
                value_size=512,
                max_keys_per_request=128,
                num_threads=4,
                dram_allocator="PERSIST_LOOP_SLAB",
                local_shm_region="unused",
                local_shm_slot_count=64,
                local_shm_ready_queue_count=1,
                local_shm_ready_queue_burst_limit=8,
                local_shm_slot_buffer_bytes=8192,
                local_shm_client_timeout_ms=1000,
                dram_capacity_multiplier=2.0,
            )
            config_path = tmp_path / "rdma.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            runner = MagicMock()
            runner.build_server_cmd.return_value = ["/tmp/petps_server", "--server"]
            runner.run.return_value.__enter__.return_value = runner
            runner.run.return_value.__exit__.return_value = None
            runner.run_client.return_value = SimpleNamespace(
                returncode=0,
                stdout=(
                    "PS_BENCHMARK_RESULT phase=run transport=RDMA mode=fetch "
                    "distribution=uniform zipfian_alpha=0.9 threads=1 batch_size=16 "
                    "records=1024 runtime_s=1 batches=1 key_ops=16 "
                    "throughput_batches_sec=1 throughput_keys_sec=16\n"
                ),
                stderr="",
            )

            with patch(
                "run_ps_dram_transport_benchmark.PetPSClusterRunner",
                return_value=runner,
            ) as runner_cls:
                outputs = run_one_case(
                    repo_root=Path("/repo"),
                    server_binary=Path("/tmp/petps_server"),
                    benchmark_binary=Path("/tmp/ps_transport_benchmark"),
                    config_path=config_path,
                    server_log_path=tmp_path / "server.log",
                    transport="RDMA",
                    num_shards=2,
                    mode="fetch",
                    record_count=1024,
                    runtime_seconds=1,
                    threads=1,
                    load_threads=1,
                    batch_size=16,
                    value_size=512,
                    distribution="uniform",
                    zipfian_alpha=0.9,
                    read_ratio=100,
                    report_mode="summary",
                    startup_delay=0.1,
                    client_timeout_s=10,
                    rdma_thread_num=3,
                    rdma_use_local_memcached="auto",
                    rdma_memcached_host="127.0.0.1",
                    rdma_memcached_port=21211,
                    rdma_transport_mode="descriptor_doorbell",
                    rdma_put_protocol_version=2,
                    rdma_put_v2_transfer_mode="read",
                    rdma_wait_timeout_ms=30000,
                    rdma_server_ready_timeout_sec=30,
                    rdma_client_receive_arena_bytes=0,
                    rdma_put_client_send_arena_bytes=0,
                    rdma_per_thread_response_limit_bytes=1048576,
                    rdma_put_server_scratch_bytes=0,
                    rdma_put_v2_push_slot_bytes=0,
                    rdma_put_v2_push_slots_per_client=0,
                    rdma_put_v2_push_region_offset=0,
                    show_runner_logs=False,
                    client_processes=1,
                )

        self.assertEqual(len(outputs), 1)
        _, stdout, stderr = outputs[0]
        self.assertIn("transport=RDMA", stdout)
        self.assertEqual(stderr, "")
        _, kwargs = runner_cls.call_args
        self.assertEqual(kwargs["server_path"], Path("/tmp/petps_server"))
        self.assertEqual(kwargs["config_path"], config_path)
        self.assertEqual(kwargs["num_servers"], 2)
        self.assertEqual(kwargs["thread_num"], 3)
        self.assertEqual(kwargs["value_size"], 512)
        self.assertEqual(kwargs["max_kv_num_per_request"], 128)
        self.assertEqual(kwargs["rdma_transport_mode"], "descriptor_doorbell")
        self.assertEqual(kwargs["rdma_put_v2_transfer_mode"], "read")
        self.assertEqual(kwargs["rdma_per_thread_response_limit_bytes"], 1048576)
        self.assertTrue(kwargs["rdma_use_dram"])
        client_cmd = runner.run_client.call_args.args[0]
        self.assertEqual(runner.run_client.call_args.kwargs["client_index"], 0)
        self.assertIn("--transport=rdma", client_cmd)
        self.assertIn("--num_shards=2", client_cmd)
        self.assertIn(f"--config_path={config_path}", client_cmd)

    def test_local_shm_config_contains_transport_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_runtime_config(
                transport="LOCAL_SHM",
                index_type="DRAM_EXTENDIBLE_HASH",
                runtime_dir=Path(tmpdir),
                num_shards=1,
                base_port=0,
                capacity=1024,
                value_size=64,
                max_keys_per_request=128,
                num_threads=1,
                dram_allocator="PERSIST_LOOP_SLAB",
                local_shm_region="bench_region",
                local_shm_slot_count=32,
                local_shm_ready_queue_count=2,
                local_shm_ready_queue_burst_limit=4,
                local_shm_slot_buffer_bytes=8192,
                local_shm_client_timeout_ms=2000,
                dram_capacity_multiplier=2.0,
            )
        self.assertEqual(config["cache_ps"]["ps_type"], "LOCAL_SHM")
        self.assertEqual(config["local_shm"]["region_name"], "bench_region")
        self.assertEqual(config["local_shm"]["ready_queue_count"], 2)

    def test_build_benchmark_cmd_includes_transport_and_value_size(self):
        cmd = build_benchmark_cmd(
            benchmark_binary=Path("/tmp/ps_transport_benchmark"),
            transport="GRPC",
            host="127.0.0.1",
            port=15100,
            num_shards=2,
            config_path=Path("/tmp/config.json"),
            mode="fetch",
            record_count=1000,
            runtime_seconds=5,
            threads=16,
            load_threads=0,
            batch_size=64,
            value_size=256,
            distribution="uniform",
            zipfian_alpha=0.9,
            read_ratio=100,
            report_mode="summary",
        )
        self.assertEqual(cmd[0], "/tmp/ps_transport_benchmark")
        self.assertIn("--transport=grpc", cmd)
        self.assertIn("--port=15100", cmd)
        self.assertIn("--config_path=/tmp/config.json", cmd)
        self.assertIn("--workload=transactions", cmd)
        self.assertIn("--value_size=256", cmd)

    def test_collect_summary_rows_keeps_measure_rows(self):
        sample = (
            "transport=GRPC op=put phase=warmup summary rounds=1 iterations=10 "
            "batch_keys=64 elapsed_us_mean=200 elapsed_us_p50=200 "
            "elapsed_us_p95=200 elapsed_us_p99=200 ops_per_sec=100 key_ops_per_sec=6400\n"
            "transport=GRPC op=get phase=measure summary rounds=3 iterations=10 "
            "batch_keys=64 elapsed_us_mean=100 elapsed_us_p50=90 "
            "elapsed_us_p95=120 elapsed_us_p99=130 ops_per_sec=200 key_ops_per_sec=12800\n"
        )
        rows = collect_summary_rows(sample)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["transport"], "GRPC")
        self.assertEqual(rows[0]["op"], "get")
        self.assertEqual(rows[0]["batch_keys"], 64)

    def test_print_summary_table_renders_index_type(self):
        out = StringIO()
        with redirect_stdout(out):
            print_summary_table(
                [
                    {
                        "index_type": "DRAM_EXTENDIBLE_HASH",
                        "transport": "BRPC",
                        "mode": "fetch",
                        "phase": "run",
                        "threads": 16,
                        "batch_size": 1024,
                        "records": 1000000,
                        "throughput_keys_sec": 12800000.0,
                    }
                ]
            )
        text = out.getvalue()
        self.assertIn("PS DRAM Transport Benchmark Summary", text)
        self.assertIn("DRAM_EXTENDIBLE_HASH", text)
        self.assertIn("BRPC", text)

    def test_write_csv_writes_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "out.csv"
            write_csv(
                [
                    {
                        "index_type": "DRAM_UNORDERED_MAP",
                        "value_store_type": "DRAM_VALUE_STORE",
                        "value_size": 512,
                        "capacity": 1024,
                        "transport": "LOCAL_SHM",
                        "phase": "measure",
                        "mode": "fetch",
                        "read_ratio": 100,
                        "threads": 16,
                        "batch_size": 1024,
                        "records": 1024,
                        "distribution": "uniform",
                        "zipfian_alpha": 0.9,
                        "runtime_s": 5.0,
                        "batches": 1,
                        "key_ops": 1024,
                        "throughput_batches_sec": 1.0,
                        "throughput_keys_sec": 1024.0,
                    }
                ],
                csv_path,
            )
            rows = csv_path.read_text(encoding="utf-8")
        self.assertIn("index_type,value_store_type", rows)
        self.assertIn("DRAM_UNORDERED_MAP,DRAM_VALUE_STORE", rows)

    def test_config_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_runtime_config(
                transport="GRPC",
                index_type="DRAM_UNORDERED_MAP",
                runtime_dir=Path(tmpdir),
                num_shards=2,
                base_port=15100,
                capacity=1024,
                value_size=64,
                max_keys_per_request=128,
                num_threads=2,
                dram_allocator="PERSIST_LOOP_SLAB",
                local_shm_region="unused",
                local_shm_slot_count=64,
                local_shm_ready_queue_count=1,
                local_shm_ready_queue_burst_limit=8,
                local_shm_slot_buffer_bytes=8192,
                local_shm_client_timeout_ms=1000,
                dram_capacity_multiplier=2.0,
            )
            loaded = json.loads(json.dumps(config))
        self.assertEqual(loaded["cache_ps"]["base_kv_config"]["capacity"], 1024)

    def test_parse_csv_list_normalizes_values(self):
        self.assertEqual(parse_csv_list("grpc, BRPC, rdma"), ["GRPC", "BRPC", "RDMA"])

    def test_collect_ps_result_rows_parses_transactions(self):
        text = (
            "PS_BENCHMARK_RESULT phase=run transport=BRPC mode=fetch "
            "distribution=uniform zipfian_alpha=0.9 threads=16 batch_size=1024 "
            "records=1000000 runtime_s=5.0 batches=10 key_ops=10240 "
            "throughput_batches_sec=2 throughput_keys_sec=2048\n"
        )
        rows = collect_ps_result_rows(text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["transport"], "BRPC")
        self.assertEqual(rows[0]["threads"], 16)
        self.assertEqual(rows[0]["throughput_keys_sec"], 2048.0)

    def test_collect_case_rows_adds_process_metadata_and_aggregate(self):
        outputs = [
            (
                0,
                "PS_BENCHMARK_RESULT phase=run transport=BRPC mode=fetch "
                "distribution=uniform zipfian_alpha=0.9 threads=16 batch_size=1024 "
                "records=1000000 runtime_s=5.0 batches=10 key_ops=10240 "
                "throughput_batches_sec=2 throughput_keys_sec=2048\n",
            ),
            (
                1,
                "PS_BENCHMARK_RESULT phase=run transport=BRPC mode=fetch "
                "distribution=uniform zipfian_alpha=0.9 threads=16 batch_size=1024 "
                "records=1000000 runtime_s=5.0 batches=20 key_ops=20480 "
                "throughput_batches_sec=4 throughput_keys_sec=4096\n",
            ),
        ]
        rows = collect_case_rows(
            outputs,
            index_type="DRAM_PET_HASH",
            value_size=512,
            capacity=1000000,
            read_ratio=100,
            client_processes=2,
        )
        per_process = [row for row in rows if row["aggregate"] == "false"]
        aggregate = [row for row in rows if row["aggregate"] == "true"]
        self.assertEqual(len(per_process), 2)
        self.assertEqual(len(aggregate), 1)
        self.assertEqual(per_process[1]["process_id"], 1)
        self.assertEqual(aggregate[0]["process_id"], "all")
        self.assertEqual(aggregate[0]["threads"], 32)
        self.assertEqual(aggregate[0]["client_processes"], 2)
        self.assertEqual(aggregate[0]["throughput_keys_sec"], 6144.0)

    def test_is_port_open_returns_false_for_unused_port(self):
        self.assertFalse(is_port_open("127.0.0.1", 1, timeout_s=0.01))


if __name__ == "__main__":
    unittest.main()
