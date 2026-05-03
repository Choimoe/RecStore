#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HIDDEN_ENGINES = {"basic"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render YCSB CI Pages artifacts.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing-history", type=Path, default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--workflow-url", default="")
    parser.add_argument("--record-count", type=int, default=1000)
    parser.add_argument("--operation-count", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-history", type=int, default=200)
    return parser


def _coerce_value(value: str) -> int | float | str:
    if value == "":
        return ""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_summary(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({key: _coerce_value(value) for key, value in row.items()})
    return rows


def read_history(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_history(path: Path, entries: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("workload", "")), str(row.get("engine", ""))


def _row_succeeded(row: dict[str, Any]) -> bool:
    exit_code = row.get("exit_code", 1)
    return int(exit_code if exit_code != "" else 1) == 0


def _workload_note(workloads: list[str]) -> str:
    notes = {
        "workloada": "workloada: run 阶段是 50% read + 50% update 的混合负载。",
        "workloadb": "workloadb: run 阶段以读为主，约 95% read + 5% update。",
        "workloadc": "workloadc: run 阶段是 100% read。",
        "workloadd": "workloadd: run 阶段以 latest 分布读为主，并包含 insert。",
        "workloade": "workloade: run 阶段包含 scan 和 insert。",
        "workloadf": "workloadf: run 阶段包含 read-modify-write 和 read。",
        "workload-update": "workload-update: run 阶段是 100% update。",
    }
    return " ".join(notes.get(workload, f"{workload}: run 阶段按该 YCSB workload 定义执行。") for workload in workloads)


def _workload_title(workload: str) -> str:
    titles = {
        "workloada": "workloada · 50% read / 50% update",
        "workloadc": "workloadc · 100% read",
        "workload-update": "workload-update · 100% update",
    }
    return titles.get(workload, workload)


def _history_series(
    history: list[dict[str, Any]], workload: str, engine: str
) -> list[tuple[str, float]]:
    points: list[tuple[str, float]] = []
    for index, entry in enumerate(history):
        run = entry.get("run", {})
        label = str(run.get("run_id") or run.get("sha") or index)
        for row in entry.get("rows", []):
            if _row_key(row) == (workload, engine) and _row_succeeded(row):
                points.append((label, float(row.get("run_throughput_ops_sec", 0) or 0)))
                break
    return points


def render_history_chart(history: list[dict[str, Any]], latest_rows: list[dict[str, Any]]) -> str:
    keys = [_row_key(row) for row in latest_rows if _row_succeeded(row)]
    series = [(_history_series(history, workload, engine), workload, engine) for workload, engine in keys]
    series = [(points, workload, engine) for points, workload, engine in series if points]
    if not series:
        return "<p class=\"muted\">暂无可展示的历史趋势。</p>"

    width = 760
    height = 280
    left = 56
    right = 24
    top = 18
    bottom = 42
    values = [value for points, _, _ in series for _, value in points]
    max_value = max(values) if values else 1.0
    max_points = max(len(points) for points, _, _ in series)
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c", "#0891b2", "#4b5563"]

    def x_for(index: int, count: int) -> float:
        if count <= 1:
            return left + plot_width
        return left + (plot_width * index / (count - 1))

    def y_for(value: float) -> float:
        if max_value <= 0:
            return top + plot_height
        return top + plot_height - (plot_height * value / max_value)

    parts = [
        f'<svg class="trend-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Run throughput history trend">',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" class="axis" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis" />',
        f'<text x="{left}" y="{top + plot_height + 28}" class="axis-label">older runs</text>',
        f'<text x="{width - right - 72}" y="{top + plot_height + 28}" class="axis-label">latest run</text>',
        f'<text x="8" y="{top + 10}" class="axis-label">ops/s</text>',
        f'<text x="{left + 4}" y="{top + 14}" class="axis-label">max {html.escape(format_number(max_value))}</text>',
    ]
    for offset, (points, workload, engine) in enumerate(series):
        color = colors[offset % len(colors)]
        coordinates = [
            f"{x_for(index, len(points)):.1f},{y_for(value):.1f}"
            for index, (_, value) in enumerate(points)
        ]
        label = f"{engine} / {workload}"
        if len(coordinates) == 1:
            x, y = coordinates[0].split(",")
            parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"><title>{html.escape(label)}</title></circle>')
        else:
            parts.append(
                f'<polyline points="{" ".join(coordinates)}" fill="none" stroke="{color}" stroke-width="2.4">'
                f"<title>{html.escape(label)}</title></polyline>"
            )
            for coordinate in coordinates[-3:]:
                x, y = coordinate.split(",")
                parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}" />')
    parts.append("</svg>")

    legend = []
    for offset, (_, workload, engine) in enumerate(series):
        color = colors[offset % len(colors)]
        legend.append(
            f'<span class="legend-item"><i style="background:{color}"></i>{html.escape(engine)} / {html.escape(workload)}</span>'
        )
    return "\n".join(parts) + f'<div class="legend">{"".join(legend)}</div>'


def render_delta_table(history: list[dict[str, Any]], latest_rows: list[dict[str, Any]]) -> str:
    if len(history) < 2:
        return "<p class=\"muted\">需要至少两次发布后的历史记录才能计算变化。</p>"

    rows = []
    for row in latest_rows:
        workload, engine = _row_key(row)
        if not _row_succeeded(row):
            continue
        points = _history_series(history, workload, engine)
        if len(points) < 2:
            continue
        previous = points[-2][1]
        current = points[-1][1]
        delta = current - previous
        percent = "" if previous == 0 else f"{delta * 100 / previous:+.2f}%"
        rows.append(
            "<tr>"
            f"<td>{html.escape(workload)}</td>"
            f"<td>{html.escape(engine)}</td>"
            f"<td>{html.escape(format_number(previous))}</td>"
            f"<td>{html.escape(format_number(current))}</td>"
            f"<td>{html.escape(format_number(delta))}</td>"
            f"<td>{html.escape(percent)}</td>"
            "</tr>"
        )
    if not rows:
        return "<p class=\"muted\">当前历史记录不足以比较相同 workload/engine 的相邻变化。</p>"
    return (
        "<table>"
        "<thead><tr><th>Workload</th><th>Engine</th><th>Previous run ops/s</th><th>Latest run ops/s</th><th>Delta ops/s</th><th>Delta %</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def render_table(rows: list[dict[str, Any]]) -> str:
    cells = []
    for row in rows:
        cells.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('workload', '')))}</td>"
            f"<td>{html.escape(str(row.get('engine', '')))}</td>"
            f"<td>{html.escape(format_number(row.get('load_throughput_ops_sec', '')))}</td>"
            f"<td>{html.escape(format_number(row.get('run_throughput_ops_sec', '')))}</td>"
            f"<td>{html.escape(str(row.get('exit_code', '')))}</td>"
            "</tr>"
        )
    return "\n".join(cells)


def render_bar_chart(rows: list[dict[str, Any]]) -> str:
    values = [
        float(row.get("run_throughput_ops_sec", 0) or 0)
        for row in rows
        if _row_succeeded(row)
    ]
    max_value = max(values) if values else 1.0
    bars = []
    for row in rows:
        value = float(row.get("run_throughput_ops_sec", 0) or 0)
        width = 2 if max_value <= 0 else max(2, int(value * 100 / max_value))
        label = str(row.get("engine", ""))
        bars.append(
            "<div class=\"bar-row\">"
            f"<span>{html.escape(label)}</span>"
            f"<div class=\"bar\"><i style=\"width:{width}%\"></i></div>"
            f"<b>{html.escape(format_number(value))}</b>"
            "</div>"
        )
    return "\n".join(bars)


def render_workload_panels(
    workloads: list[str],
    rows_by_workload: dict[str, list[dict[str, Any]]],
    history: list[dict[str, Any]],
) -> str:
    buttons = []
    panels = []
    for index, workload in enumerate(workloads):
        active = index == 0
        tab_id = f"tab-{index}"
        panel_id = f"panel-{index}"
        workload_rows = rows_by_workload[workload]
        passed = sum(1 for row in workload_rows if _row_succeeded(row))
        buttons.append(
            f'<button class="tab-button{" active" if active else ""}" id="{tab_id}" '
            f'role="tab" aria-selected="{str(active).lower()}" aria-controls="{panel_id}" '
            f'data-tab-target="{panel_id}" type="button">{html.escape(_workload_title(workload))}</button>'
        )
        panels.append(
            f'<section class="tab-panel{" active" if active else ""}" id="{panel_id}" '
            f'role="tabpanel" aria-labelledby="{tab_id}">'
            '<div class="grid">'
            "<div>"
            "<h3>Latest Run Throughput</h3>"
            '<p class="muted">主指标是 YCSB <b>run</b> 阶段吞吐，单位 ops/s；load 阶段是初始装载写入。</p>'
            f"{render_bar_chart(workload_rows)}"
            "</div>"
            "<div>"
            "<h3>History Trend</h3>"
            '<p class="muted">横轴从旧到新，纵轴为 run ops/s。</p>'
            f"{render_history_chart(history, workload_rows)}"
            "</div>"
            "</div>"
            "<h3>Latest Change</h3>"
            '<p class="muted">比较当前发布与上一条相同 engine 的 run 阶段吞吐。</p>'
            f"{render_delta_table(history, workload_rows)}"
            "<h3>Latest Result Table</h3>"
            "<table>"
            "<thead><tr><th>Workload</th><th>Engine</th><th>Load ops/s</th><th>Run ops/s</th><th>Exit</th></tr></thead>"
            f"<tbody>{render_table(workload_rows)}</tbody>"
            "</table>"
            "</section>"
        )
    return (
        '<div class="tabs" role="tablist" aria-label="YCSB workload tabs">'
        + "".join(buttons)
        + "</div>"
        + "".join(panels)
    )


def render_html(latest: dict[str, Any], history: list[dict[str, Any]]) -> str:
    return """<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RecStore YCSB CI Dashboard</title>
  <script>
    const savedTheme = localStorage.getItem("recstore-ycsb-theme");
    if (savedTheme === "light" || savedTheme === "dark") {
      document.documentElement.dataset.theme = savedTheme;
    }
  </script>
  <style>
    :root { color-scheme: light; --bg:#ffffff; --text:#111827; --surface:#ffffff; --subtle:#f6f8fa; --border:#d8dee4; --muted:#667085; --accent:#2563eb; }
    @media (prefers-color-scheme: dark) {
      :root:not([data-theme]) { color-scheme: dark; --bg:#0f172a; --text:#e5e7eb; --surface:#111827; --subtle:#1f2937; --border:#334155; --muted:#9ca3af; --accent:#60a5fa; }
    }
    :root[data-theme="dark"] { color-scheme: dark; --bg:#0f172a; --text:#e5e7eb; --surface:#111827; --subtle:#1f2937; --border:#334155; --muted:#9ca3af; --accent:#60a5fa; }
    :root[data-theme="light"] { color-scheme: light; --bg:#ffffff; --text:#111827; --surface:#ffffff; --subtle:#f6f8fa; --border:#d8dee4; --muted:#667085; --accent:#2563eb; }
    body { margin:0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height:1.45; background:var(--bg); color:var(--text); }
    header { padding:24px 32px; border-bottom:1px solid var(--border); }
    .header-row { display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }
    .theme-toggle { border:1px solid var(--border); background:var(--surface); color:var(--text); border-radius:8px; padding:8px 10px; cursor:pointer; font:inherit; white-space:nowrap; }
    .theme-toggle:hover { border-color:var(--accent); color:var(--accent); }
    main { max-width:1180px; margin:0 auto; padding:24px 20px 48px; }
    h1 { margin:0 0 8px; font-size:28px; }
    h2 { margin:28px 0 12px; font-size:20px; }
    h3 { margin:20px 0 10px; font-size:16px; }
    .muted { color:var(--muted); }
    .cards { display:grid; grid-template-columns:repeat(6, minmax(0, 1fr)); gap:12px; }
    .card { border:1px solid var(--border); border-radius:8px; padding:14px; background:var(--surface); }
    .card b { display:block; font-size:18px; margin-top:4px; overflow-wrap:anywhere; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; align-items:start; }
    table { width:100%; border-collapse:collapse; border:1px solid var(--border); }
    th, td { padding:8px 10px; border-bottom:1px solid var(--border); text-align:left; }
    th { background:var(--subtle); }
    .bar-row { display:grid; grid-template-columns:150px 1fr 90px; gap:10px; align-items:center; margin:8px 0; }
    .bar { height:12px; background:var(--subtle); border-radius:999px; overflow:hidden; }
    .bar i { display:block; height:100%; background:var(--accent); }
    .trend-chart { width:100%; height:auto; border:1px solid var(--border); border-radius:8px; }
    .axis { stroke:var(--border); stroke-width:1; }
    .axis-label { fill:var(--muted); font-size:12px; }
    .legend { display:flex; flex-wrap:wrap; gap:10px 16px; margin-top:10px; }
    .legend-item { display:inline-flex; gap:6px; align-items:center; color:var(--muted); font-size:13px; }
    .legend-item i { width:10px; height:10px; border-radius:2px; display:inline-block; }
    .tabs { display:flex; gap:8px; flex-wrap:wrap; border-bottom:1px solid var(--border); margin-top:18px; }
    .tab-button { appearance:none; border:1px solid var(--border); border-bottom:0; background:transparent; color:inherit; padding:9px 12px; border-radius:8px 8px 0 0; cursor:pointer; font:inherit; }
    .tab-button.active { background:var(--subtle); color:var(--accent); font-weight:600; }
    .tab-panel { display:none; padding-top:8px; }
    .tab-panel.active { display:block; }
    .meta-line { display:flex; flex-wrap:wrap; gap:8px 16px; margin-top:12px; font-size:13px; }
    .status { padding:24px 0; }
    a { color:var(--accent); }
    @media (max-width: 980px) { .cards { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 820px) { .cards, .grid { grid-template-columns:1fr; } .bar-row { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <div class="header-row">
      <div>
        <h1>RecStore YCSB CI Dashboard</h1>
      </div>
      <button class="theme-toggle" type="button" id="theme-toggle" aria-label="Toggle color theme">Dark</button>
    </div>
  </header>
  <main>
    <section class="cards" id="summary-cards"></section>
    <div class="meta-line muted" id="run-meta"></div>
    <section>
      <h2>Workload Views</h2>
      <div id="workload-views" class="status muted">Loading YCSB history...</div>
    </section>
  </main>
  <script>
    const themeToggle = document.getElementById("theme-toggle");
    const systemPrefersDark = () => window.matchMedia("(prefers-color-scheme: dark)").matches;
    const currentTheme = () => document.documentElement.dataset.theme || (systemPrefersDark() ? "dark" : "light");
    const syncThemeToggle = () => {
      themeToggle.textContent = currentTheme() === "dark" ? "Light" : "Dark";
    };
    themeToggle.addEventListener("click", () => {
      const nextTheme = currentTheme() === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = nextTheme;
      localStorage.setItem("recstore-ycsb-theme", nextTheme);
      syncThemeToggle();
    });
    syncThemeToggle();

    const hiddenEngines = new Set(["basic"]);
    const workloadTitle = (workload) => ({
      "workloada": "workloada · 50% read / 50% update",
      "workloadc": "workloadc · 100% read",
      "workload-update": "workload-update · 100% update"
    }[workload] || workload);
    const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[ch]));
    const number = (value) => typeof value === "number" ? value.toFixed(2) : String(value ?? "");
    const rowOk = (row) => Number(row.exit_code ?? 1) === 0;
    const visibleRows = (rows) => rows.filter((row) => !hiddenEngines.has(String(row.engine || "")));

    async function loadData() {
      const [latestResponse, historyResponse] = await Promise.all([
        fetch("latest/run.json", { cache: "no-store" }),
        fetch("history.jsonl", { cache: "no-store" })
      ]);
      if (!latestResponse.ok) throw new Error("latest/run.json not found");
      const latest = await latestResponse.json();
      const historyText = historyResponse.ok ? await historyResponse.text() : "";
      const history = historyText.split("\\n").map((line) => line.trim()).filter(Boolean).map(JSON.parse);
      return { latest, history };
    }

    function renderCards(latest, rows) {
      const run = latest.run || {};
      const engines = new Set(rows.map((row) => String(row.engine || "")));
      const workloads = new Set(rows.map((row) => String(row.workload || "")));
      const passed = rows.filter(rowOk).length;
      const cards = [
        ["Commit", String(run.sha || "").slice(0, 12)],
        ["Run ID", run.run_id || ""],
        ["Branch", run.branch || ""],
        ["Engines", engines.size],
        ["Workloads", workloads.size],
        ["Passed Rows", `${passed}/${rows.length}`]
      ];
      document.getElementById("summary-cards").innerHTML = cards.map(([label, value]) =>
        `<div class="card"><span class="muted">${esc(label)}</span><b>${esc(value)}</b></div>`
      ).join("");
      document.getElementById("run-meta").innerHTML = [
        `Records ${esc(run.record_count || "")}`,
        `Operations ${esc(run.operation_count || "")}`,
        `Threads ${esc(run.threads || "")}`,
        `Generated ${esc(run.generated_at || "")}`,
        `<a href="latest/summary.csv">summary.csv</a>`
      ].map((item) => `<span>${item}</span>`).join("");
    }

    function historySeries(history, workload, engine) {
      const points = [];
      for (const [index, entry] of history.entries()) {
        const row = visibleRows(entry.rows || []).find((item) =>
          String(item.workload || "") === workload && String(item.engine || "") === engine && rowOk(item)
        );
        if (row) {
          const label = entry.run?.run_id || entry.run?.sha || String(index);
          points.push([label, Number(row.run_throughput_ops_sec || 0)]);
        }
      }
      return points;
    }

    function renderBars(rows) {
      const values = rows.filter(rowOk).map((row) => Number(row.run_throughput_ops_sec || 0));
      const maxValue = Math.max(1, ...values);
      return rows.map((row) => {
        const value = Number(row.run_throughput_ops_sec || 0);
        const width = Math.max(2, Math.floor(value * 100 / maxValue));
        return `<div class="bar-row"><span>${esc(row.engine)}</span><div class="bar"><i style="width:${width}%"></i></div><b>${esc(number(value))}</b></div>`;
      }).join("");
    }

    function renderTrend(history, rows) {
      const series = rows.filter(rowOk).map((row) => ({
        workload: String(row.workload || ""),
        engine: String(row.engine || ""),
        points: historySeries(history, String(row.workload || ""), String(row.engine || ""))
      })).filter((item) => item.points.length);
      if (!series.length) return '<p class="muted">暂无可展示的历史趋势。</p>';
      const width = 760, height = 280, left = 56, right = 24, top = 18, bottom = 42;
      const plotWidth = width - left - right;
      const plotHeight = height - top - bottom;
      const maxValue = Math.max(1, ...series.flatMap((item) => item.points.map((point) => point[1])));
      const colors = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c", "#0891b2", "#4b5563"];
      const xFor = (index, count) => count <= 1 ? left + plotWidth : left + (plotWidth * index / (count - 1));
      const yFor = (value) => top + plotHeight - (plotHeight * value / maxValue);
      let svg = `<svg class="trend-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Run throughput history trend">`;
      svg += `<line x1="${left}" y1="${top + plotHeight}" x2="${width - right}" y2="${top + plotHeight}" class="axis" />`;
      svg += `<line x1="${left}" y1="${top}" x2="${left}" y2="${top + plotHeight}" class="axis" />`;
      svg += `<text x="${left}" y="${top + plotHeight + 28}" class="axis-label">older runs</text>`;
      svg += `<text x="${width - right - 72}" y="${top + plotHeight + 28}" class="axis-label">latest run</text>`;
      svg += `<text x="8" y="${top + 10}" class="axis-label">ops/s</text>`;
      svg += `<text x="${left + 4}" y="${top + 14}" class="axis-label">max ${esc(number(maxValue))}</text>`;
      const legend = [];
      for (const [offset, item] of series.entries()) {
        const color = colors[offset % colors.length];
        const coords = item.points.map((point, index) => `${xFor(index, item.points.length).toFixed(1)},${yFor(point[1]).toFixed(1)}`);
        const label = `${item.engine} / ${item.workload}`;
        if (coords.length === 1) {
          const [x, y] = coords[0].split(",");
          svg += `<circle cx="${x}" cy="${y}" r="4" fill="${color}"><title>${esc(label)}</title></circle>`;
        } else {
          svg += `<polyline points="${coords.join(" ")}" fill="none" stroke="${color}" stroke-width="2.4"><title>${esc(label)}</title></polyline>`;
          for (const coord of coords.slice(-3)) {
            const [x, y] = coord.split(",");
            svg += `<circle cx="${x}" cy="${y}" r="3" fill="${color}" />`;
          }
        }
        legend.push(`<span class="legend-item"><i style="background:${color}"></i>${esc(label)}</span>`);
      }
      return `${svg}</svg><div class="legend">${legend.join("")}</div>`;
    }

    function renderDelta(history, rows) {
      const body = rows.filter(rowOk).map((row) => {
        const points = historySeries(history, String(row.workload || ""), String(row.engine || ""));
        if (points.length < 2) return "";
        const previous = points[points.length - 2][1];
        const current = points[points.length - 1][1];
        const delta = current - previous;
        const percent = previous === 0 ? "" : `${(delta * 100 / previous).toFixed(2)}%`;
        return `<tr><td>${esc(row.workload)}</td><td>${esc(row.engine)}</td><td>${esc(number(previous))}</td><td>${esc(number(current))}</td><td>${esc(number(delta))}</td><td>${esc(delta >= 0 && percent ? "+" + percent : percent)}</td></tr>`;
      }).join("");
      if (!body) return '<p class="muted">当前历史记录不足以比较相同 workload/engine 的相邻变化。</p>';
      return `<table><thead><tr><th>Workload</th><th>Engine</th><th>Previous run ops/s</th><th>Latest run ops/s</th><th>Delta ops/s</th><th>Delta %</th></tr></thead><tbody>${body}</tbody></table>`;
    }

    function renderTable(rows) {
      return `<table><thead><tr><th>Workload</th><th>Engine</th><th>Load ops/s</th><th>Run ops/s</th><th>Exit</th></tr></thead><tbody>${rows.map((row) =>
        `<tr><td>${esc(row.workload)}</td><td>${esc(row.engine)}</td><td>${esc(number(row.load_throughput_ops_sec))}</td><td>${esc(number(row.run_throughput_ops_sec))}</td><td>${esc(row.exit_code)}</td></tr>`
      ).join("")}</tbody></table>`;
    }

    function activateTabs() {
      for (const button of document.querySelectorAll("[data-tab-target]")) {
        button.addEventListener("click", () => {
        const targetId = button.getAttribute("data-tab-target");
        for (const item of document.querySelectorAll(".tab-button")) {
          item.classList.toggle("active", item === button);
          item.setAttribute("aria-selected", item === button ? "true" : "false");
        }
        for (const panel of document.querySelectorAll(".tab-panel")) {
          panel.classList.toggle("active", panel.id === targetId);
        }
        });
      }
    }

    function renderWorkloads(latest, history) {
      const rows = visibleRows(latest.rows || []);
      renderCards(latest, rows);
      const workloads = [...new Set(rows.map((row) => String(row.workload || "")))].sort();
      const tabs = workloads.map((workload, index) =>
        `<button class="tab-button${index === 0 ? " active" : ""}" id="tab-${index}" role="tab" aria-selected="${index === 0 ? "true" : "false"}" aria-controls="panel-${index}" data-tab-target="panel-${index}" type="button">${esc(workloadTitle(workload))}</button>`
      ).join("");
      const panels = workloads.map((workload, index) => {
        const workloadRows = rows.filter((row) => String(row.workload || "") === workload);
        return `<section class="tab-panel${index === 0 ? " active" : ""}" id="panel-${index}" role="tabpanel" aria-labelledby="tab-${index}"><div class="grid"><div><h3>Latest Run Throughput</h3><p class="muted">主指标是 YCSB <b>run</b> 阶段吞吐，单位 ops/s；load 阶段是初始装载写入。</p>${renderBars(workloadRows)}</div><div><h3>History Trend</h3><p class="muted">横轴从旧到新，纵轴为 run ops/s。</p>${renderTrend(history, workloadRows)}</div></div><h3>Latest Change</h3><p class="muted">比较当前发布与上一条相同 engine 的 run 阶段吞吐。</p>${renderDelta(history, workloadRows)}<h3>Latest Result Table</h3>${renderTable(workloadRows)}</section>`;
      }).join("");
      document.getElementById("workload-views").classList.remove("status", "muted");
      document.getElementById("workload-views").innerHTML = `<div class="tabs" role="tablist" aria-label="YCSB workload tabs">${tabs}</div>${panels}`;
      activateTabs();
    }

    loadData().then(({ latest, history }) => renderWorkloads(latest, history)).catch((error) => {
      document.getElementById("workload-views").textContent = `Failed to load YCSB data: ${error.message}`;
    });
  </script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = read_summary(args.summary_csv)
    output_dir = args.output_dir
    latest_dir = output_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    run = {
        "run_id": args.run_id,
        "sha": args.sha,
        "branch": args.branch,
        "workflow_url": args.workflow_url,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": args.record_count,
        "operation_count": args.operation_count,
        "threads": args.threads,
    }
    latest = {"run": run, "rows": rows}
    history = read_history(args.existing_history)
    history.append(latest)
    history = history[-args.max_history :]

    shutil.copyfile(args.summary_csv, latest_dir / "summary.csv")
    write_json(latest_dir / "run.json", latest)
    write_history(output_dir / "history.jsonl", history)
    (output_dir / "index.html").write_text(
        render_html(latest, history), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
