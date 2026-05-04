const savedTheme = localStorage.getItem("recstore-ycsb-theme");
if (savedTheme === "light" || savedTheme === "dark") {
  document.documentElement.dataset.theme = savedTheme;
}

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
  "workload-update": "workload-update · 100% update",
}[workload] || workload);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
}[ch]));
const number = (value) => typeof value === "number" ? value.toFixed(2) : String(value ?? "");
const rowOk = (row) => Number(row.exit_code ?? 1) === 0;
const visibleRows = (rows) => rows.filter((row) => !hiddenEngines.has(String(row.engine || "")));

async function loadData() {
  const [latestResponse, historyResponse] = await Promise.all([
    fetch("latest/run.json", { cache: "no-store" }),
    fetch("history.jsonl", { cache: "no-store" }),
  ]);
  if (!latestResponse.ok) throw new Error("latest/run.json not found");
  const latest = await latestResponse.json();
  const historyText = historyResponse.ok ? await historyResponse.text() : "";
  const history = historyText.split("\n").map((line) => line.trim()).filter(Boolean).map(JSON.parse);
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
    ["Passed Rows", `${passed}/${rows.length}`],
  ];
  document.getElementById("summary-cards").innerHTML = cards.map(([label, value]) =>
    `<div class="card"><span class="muted">${esc(label)}</span><b>${esc(value)}</b></div>`
  ).join("");
  document.getElementById("run-meta").innerHTML = [
    `Records ${esc(run.record_count || "")}`,
    `Operations ${esc(run.operation_count || "")}`,
    `Threads ${esc(run.threads || "")}`,
    `Generated ${esc(run.generated_at || "")}`,
    '<a href="latest/summary.csv">summary.csv</a>',
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
    points: historySeries(history, String(row.workload || ""), String(row.engine || "")),
  })).filter((item) => item.points.length);
  if (!series.length) return '<p class="muted">暂无可展示的历史趋势。</p>';
  const width = 760;
  const height = 280;
  const left = 56;
  const right = 24;
  const top = 18;
  const bottom = 42;
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
