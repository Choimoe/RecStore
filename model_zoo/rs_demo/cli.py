#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from .config import RunConfig, ensure_parent_dirs, parse_config
from .runtime.report import analyze_embupdate, setup_local_report_env
from .runtime.server import (
    choose_available_ports,
    make_runtime_dir,
    resolve_default_ports,
    start_server,
    stop_server,
    wait_server_ready,
)
from .runners.recstore_runner import RecStoreRunner


def repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


def build_runner(cfg: RunConfig, runtime_dir: Path):
    if cfg.backend == "recstore":
        return RecStoreRunner(runtime_dir)
    if cfg.backend == "torchrec":
        try:
            from .runners.torchrec_runner import TorchRecRunner, check_torchrec_available
        except ModuleNotFoundError as exc:
            raise RuntimeError("TorchRec runner is not available") from exc
        check_torchrec_available()
        return TorchRecRunner(runtime_dir)
    raise ValueError(f"Unsupported backend: {cfg.backend}")


def main(argv: list[str] | None = None) -> int:
    cfg = parse_config(argv)
    ensure_parent_dirs(cfg)
    setup_local_report_env(cfg.jsonl)

    if cfg.backend != "recstore":
        cfg.start_server = False

    repo_root = repo_root_from_this_file()
    with open(repo_root / "recstore_config.json", "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    p0_default, p1_default = resolve_default_ports(base_cfg)
    if cfg.server_port0 is None:
        cfg.server_port0 = p0_default
    if cfg.server_port1 is None:
        cfg.server_port1 = p1_default
    server_needed = cfg.backend == "recstore" and cfg.start_server
    if server_needed:
        cfg.server_port0, cfg.server_port1 = choose_available_ports(
            cfg.server_host, cfg.server_port0, cfg.server_port1
        )

    runtime_dir, runtime_cfg_path = make_runtime_dir(
        base_cfg=base_cfg,
        host=cfg.server_host,
        port0=cfg.server_port0,
        port1=cfg.server_port1,
        allocator=cfg.allocator,
        ps_type=cfg.ps_type,
    )

    proc = None
    try:
        if server_needed:
            print(f"[rs_demo] starting server ({cfg.ps_type}) with {runtime_cfg_path}")
            proc = start_server(repo_root, runtime_cfg_path, Path(cfg.server_log))
            if not wait_server_ready(
                proc=proc,
                host=cfg.server_host,
                port0=cfg.server_port0,
                port1=cfg.server_port1,
                timeout_s=cfg.server_wait_seconds,
            ):
                raise RuntimeError(
                    f"server failed to become ready: {cfg.server_host}:{cfg.server_port0},{cfg.server_port1}; "
                    f"log={cfg.server_log}"
                )
            print("[rs_demo] server is ready")

        runner = build_runner(cfg, runtime_dir)
        _run_result = runner.run(repo_root, cfg)
        if cfg.backend == "torchrec":
            print(f"[rs_demo] torchrec main csv: {cfg.torchrec_main_csv}")
            return 0

        print("[rs_demo] analyzing embupdate stages...")
        analyze_output = analyze_embupdate(repo_root, cfg.jsonl, cfg.csv, top_n=20)
        print(analyze_output)

        print(f"[rs_demo] jsonl: {cfg.jsonl}")
        print(f"[rs_demo] csv:   {cfg.csv}")
        if server_needed:
            print(f"[rs_demo] server log: {cfg.server_log}")
        return 0
    finally:
        stop_server(proc)


if __name__ == "__main__":
    raise SystemExit(main())
