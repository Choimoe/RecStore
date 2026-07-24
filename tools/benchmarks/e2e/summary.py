from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Iterable

from .common import SPARSE_FEATURES_PER_SAMPLE, _load_manifest, _read_csv, _to_float, _write_csv


def _warm_rows(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    return [row for row in rows if str(row.get("warmup_excluded", "0")) not in {"1", "true", "True"}]


def _mean(rows: Iterable[dict[str, str]], column: str) -> float:
    vals = [float(row[column]) for row in rows if row.get(column, "") not in {"", "nan", "NaN"}]
    if not vals:
        return 0.0
    return statistics.fmean(vals)


def _p95(rows: Iterable[dict[str, str]], column: str) -> float:
    vals = sorted(float(row[column]) for row in rows if row.get(column, "") not in {"", "nan", "NaN"})
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    idx = int(round((len(vals) - 1) * 0.95))
    return vals[idx]


def collect_e2e_summary(
    *,
    manifest: list[dict[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    del output_root
    rows_out: list[dict[str, Any]] = []
    for item in manifest:
        if item.get("status") != "ok":
            rows_out.append(
                {
                    **item,
                    "mean_step_total_ms": 0.0,
                    "p95_step_total_ms": 0.0,
                    "samples_per_sec": 0.0,
                    "lookup_mrows_per_sec": 0.0,
                    "update_mrows_per_sec": 0.0,
                }
            )
            continue
        main_csv = Path(str(item["main_csv"]))
        if not main_csv.exists():
            rows_out.append(
                {
                    **item,
                    "status": "missing_output",
                    "mean_step_total_ms": 0.0,
                    "p95_step_total_ms": 0.0,
                    "samples_per_sec": 0.0,
                    "lookup_mrows_per_sec": 0.0,
                    "update_mrows_per_sec": 0.0,
                }
            )
            continue
        warm = _warm_rows(main_csv)
        batch_size = int(item["batch_size"])
        mean_step = _mean(warm, "step_total_ms")
        mean_lookup = _mean(warm, "embed_lookup_local_ms")
        mean_update = _mean(warm, "sparse_optimizer_ms")
        samples_per_sec = (
            batch_size * 1000.0 / mean_step if mean_step > 0.0 else 0.0
        )
        sparse_rows_per_step = batch_size * SPARSE_FEATURES_PER_SAMPLE
        lookup_mrows = (
            sparse_rows_per_step / (mean_lookup / 1000.0) / 1e6
            if mean_lookup > 0.0
            else 0.0
        )
        update_mrows = (
            sparse_rows_per_step / (mean_update / 1000.0) / 1e6
            if mean_update > 0.0
            else 0.0
        )
        rows_out.append(
            {
                **item,
                "mean_step_total_ms": mean_step,
                "p95_step_total_ms": _p95(warm, "step_total_ms"),
                "mean_embed_lookup_ms": mean_lookup,
                "mean_sparse_optimizer_ms": mean_update,
                "samples_per_sec": samples_per_sec,
                "lookup_mrows_per_sec": lookup_mrows,
                "update_mrows_per_sec": update_mrows,
            }
        )
    return rows_out


def _config_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("rows", "")),
        str(row.get("batch_size", "")),
        str(row.get("num_embeddings", "")),
        str(row.get("embedding_dim", "")),
    )


def _lane_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (*_config_key(row), str(row.get("label", "")))


def _speedup(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def build_gap_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault(_config_key(row), []).append(row)

    out: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(int(v or 0) for v in item[0])):
        lane_medians: list[dict[str, Any]] = []
        lane_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        for row in rows:
            lane_groups.setdefault(_lane_key(row), []).append(row)
        for lane_rows in lane_groups.values():
            values = [_to_float(row.get("samples_per_sec")) for row in lane_rows]
            positive = [value for value in values if value > 0.0]
            if not positive:
                continue
            representative = dict(lane_rows[0])
            representative["samples_per_sec"] = statistics.median(positive)
            representative["repeat_count"] = len(positive)
            lane_medians.append(representative)

        recstore_lanes = [row for row in lane_medians if row.get("backend") == "recstore"]
        if not recstore_lanes:
            continue
        best_recstore = max(
            recstore_lanes,
            key=lambda row: _to_float(row.get("samples_per_sec")),
        )
        hbm = next(
            (
                row
                for row in lane_medians
                if row.get("backend") == "torchrec"
                and row.get("torchrec_memory_mode") == "hbm"
            ),
            None,
        )
        uvm = next(
            (
                row
                for row in lane_medians
                if row.get("backend") == "torchrec"
                and row.get("torchrec_memory_mode") == "uvm_caching"
            ),
            None,
        )
        recstore_sps = float(best_recstore.get("samples_per_sec", 0.0))
        hbm_sps = float(hbm.get("samples_per_sec", 0.0)) if hbm else 0.0
        uvm_sps = float(uvm.get("samples_per_sec", 0.0)) if uvm else 0.0
        if hbm_sps <= 0.0 and uvm_sps <= 0.0:
            continue
        out.append(
            {
                "rows": key[0],
                "batch_size": key[1],
                "num_embeddings": key[2],
                "embedding_dim": key[3],
                "best_recstore_label": best_recstore.get("label", ""),
                "best_recstore_samples_per_sec": recstore_sps,
                "torchrec_hbm_samples_per_sec": hbm_sps,
                "torchrec_uvm_samples_per_sec": uvm_sps,
                "recstore_vs_hbm_speedup": _speedup(recstore_sps, hbm_sps),
                "recstore_vs_uvm_speedup": _speedup(recstore_sps, uvm_sps),
                "best_recstore_repeat_count": best_recstore.get("repeat_count", 0),
                "torchrec_hbm_repeat_count": hbm.get("repeat_count", 0) if hbm else 0,
                "torchrec_uvm_repeat_count": uvm.get("repeat_count", 0) if uvm else 0,
            }
        )
    return out


def build_result_insights(
    *,
    summary_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    ps_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[str]:
    insights: list[str] = []
    ok_rows = [row for row in summary_rows if row.get("status") == "ok"]
    if gap_rows:
        vs_hbm = [_to_float(row.get("recstore_vs_hbm_speedup")) for row in gap_rows]
        vs_uvm = [_to_float(row.get("recstore_vs_uvm_speedup")) for row in gap_rows]
        hbm_wins = sum(1 for value in vs_hbm if value >= 1.0)
        uvm_wins = sum(1 for value in vs_uvm if value >= 1.0)
        insights.append(
            "在 {total} 个可配对配置中，最佳 RecStore 路径有 {hbm_wins} 个配置快于 TorchRec-HBM，"
            "{uvm_wins} 个配置快于 TorchRec-UVMCache；RecStore/HBM 几何均值为 {hbm_geo:.2f}x，"
            "RecStore/UVM 几何均值为 {uvm_geo:.2f}x。".format(
                total=len(gap_rows),
                hbm_wins=hbm_wins,
                uvm_wins=uvm_wins,
                hbm_geo=_geomean(vs_hbm),
                uvm_geo=_geomean(vs_uvm),
            )
        )
        small_batch = [
            _to_float(row.get("recstore_vs_uvm_speedup"))
            for row in gap_rows
            if int(row.get("batch_size", 0) or 0) <= 1024
        ]
        large_batch = [
            _to_float(row.get("recstore_vs_uvm_speedup"))
            for row in gap_rows
            if int(row.get("batch_size", 0) or 0) >= 4096
        ]
        if small_batch and large_batch:
            insights.append(
                "batch size 对结论有明显影响：batch<=1024 时 RecStore/UVM 几何均值为 "
                f"{_geomean(small_batch):.2f}x，而 batch>=4096 时为 {_geomean(large_batch):.2f}x；"
                "因此大 batch 下 TorchRec-UVM 的 GPU 侧批处理优势需要单独报告，不能被小 batch 结果掩盖。"
            )
        best = max(gap_rows, key=lambda row: _to_float(row.get("recstore_vs_uvm_speedup")))
        worst = min(gap_rows, key=lambda row: _to_float(row.get("recstore_vs_uvm_speedup")))
        insights.append(
            "相对 TorchRec-UVM 的最佳配置是 rows={rows}, batch={batch}, emb_rows={emb}, dim={dim}，"
            "速度比为 {best:.2f}x；最弱配置是 rows={wrows}, batch={wbatch}, emb_rows={wemb}, dim={wdim}，"
            "速度比为 {worst:.2f}x。".format(
                rows=best.get("rows", ""),
                batch=best.get("batch_size", ""),
                emb=best.get("num_embeddings", ""),
                dim=best.get("embedding_dim", ""),
                best=_to_float(best.get("recstore_vs_uvm_speedup")),
                wrows=worst.get("rows", ""),
                wbatch=worst.get("batch_size", ""),
                wemb=worst.get("num_embeddings", ""),
                wdim=worst.get("embedding_dim", ""),
                worst=_to_float(worst.get("recstore_vs_uvm_speedup")),
            )
        )
    prefetch_rows = [
        row for row in ok_rows
        if row.get("backend") == "recstore" and int(row.get("prefetch_depth", 0) or 0) > 0
    ]
    if prefetch_rows:
        grouped_no_prefetch: dict[tuple[str, str, str, str], float] = {}
        for row in ok_rows:
            if row.get("backend") != "recstore" or int(row.get("prefetch_depth", 0) or 0) != 0:
                continue
            key = _config_key(row)
            grouped_no_prefetch[key] = max(
                grouped_no_prefetch.get(key, 0.0),
                _to_float(row.get("samples_per_sec")),
            )
        ratios = []
        for row in prefetch_rows:
            base = grouped_no_prefetch.get(_config_key(row), 0.0)
            if base > 0.0:
                ratios.append(_to_float(row.get("samples_per_sec")) / base)
        if ratios:
            insights.append(
                "prefetch_depth>0 当前是消融项而非主结果：相对同配置最佳非 prefetch RecStore，"
                f"吞吐几何均值保留率为 {_geomean(ratios):.2f}x，应在论文中作为负例或待优化路径解释。"
            )
    run_phase = [
        row for row in ps_rows
        if row.get("status") == "ok" and str(row.get("phase", "")).lower() in {"run", "fetch", "steady"}
    ]
    if run_phase:
        mkeys = []
        for row in run_phase:
            if row.get("throughput_mkeys_sec", "") not in {"", None}:
                mkeys.append(_to_float(row.get("throughput_mkeys_sec")))
            elif row.get("key_ops_per_sec", "") not in {"", None}:
                mkeys.append(_to_float(row.get("key_ops_per_sec")) / 1e6)
        if mkeys:
            insights.append(
                f"RDMA 仅作为 PS/network 层校准：run/fetch phase 中位吞吐为 {statistics.median(mkeys):.2f} M keys/s，"
                "不能直接写入 PyTorch/model 主表或等价为端到端训练加速。"
            )
    if int(metadata.get("gpu_count", 0) or 0) < 2:
        insights.append("当前机器 GPU 数不足 2，单机多卡行只能标记 skipped；投稿前需要在 2/4/8 GPU 节点补跑扩展性。")
    return insights


def _geomean(values: Iterable[float]) -> float:
    positive = [value for value in values if value > 0.0]
    if not positive:
        return 0.0
    return float(statistics.geometric_mean(positive))


def combine_existing_roots(roots: list[Path], output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    ps_rows: list[dict[str, Any]] = []
    for root in roots:
        source_profile = root.name.replace("bench_e2e_", "").replace("paper_e2e_", "")
        for row in _load_manifest(root / "manifest.csv"):
            row.setdefault("source_profile", source_profile)
            row.setdefault("source_root", str(root))
            manifest.append(row)
        ps_path = root / "summary_ps_network.csv"
        if ps_path.exists():
            for row in _read_csv(ps_path):
                row.setdefault("source_profile", source_profile)
                row.setdefault("source_root", str(root))
                ps_rows.append(row)
    if not manifest:
        raise RuntimeError("no manifest rows found in --combine-roots inputs")
    manifest = _dedupe_manifest_for_report(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "manifest.csv", manifest)
    if ps_rows:
        _write_csv(output_root / "summary_ps_network.csv", ps_rows)
    return manifest, ps_rows


def _manifest_config_key(row: dict[str, Any]) -> tuple[str, ...]:
    config_fields = (
        "slug",
        "rows",
        "batch_size",
        "num_embeddings",
        "embedding_dim",
        "nproc_per_node",
        "repeat",
    )
    if not all(str(row.get(field, "")) for field in config_fields):
        return (str(row.get("run_id", "")),)
    return (
        str(row.get("slug", "")),
        str(row.get("backend", "")),
        str(row.get("ps_type", "")),
        str(row.get("recstore_index_type", "")),
        str(row.get("ps_kv_backend", "")),
        str(row.get("torchrec_memory_mode", "")),
        str(row.get("prefetch_depth", "")),
        str(row.get("rows", "")),
        str(row.get("batch_size", "")),
        str(row.get("num_embeddings", "")),
        str(row.get("embedding_dim", "")),
        str(row.get("nproc_per_node", "")),
        str(row.get("repeat", "")),
    )


def _dedupe_manifest_for_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep a repaired successful rerun instead of an older failed duplicate."""
    selected: dict[tuple[str, ...], dict[str, Any]] = {}
    order: list[tuple[str, ...]] = []
    for row in rows:
        key = _manifest_config_key(row)
        if key not in selected:
            selected[key] = row
            order.append(key)
            continue
        current = selected[key]
        row_ok = row.get("status") == "ok"
        current_ok = current.get("status") == "ok"
        if row_ok and not current_ok:
            selected[key] = row
        elif row_ok == current_ok:
            selected[key] = row
    return [selected[key] for key in order]
