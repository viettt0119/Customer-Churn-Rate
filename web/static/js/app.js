"use strict";

const $ = (id) => document.getElementById(id);
const COLORS = { blue: "#3987e5", orange: "#d95926", red: "#e66767", ink: "#f2f4f8", ink2: "#c3c9d6", ink3: "#98a1b3", grid: "rgba(255,255,255,0.10)", track: "rgba(255,255,255,0.10)" };
const FEATURES = {
  tenure: "Số tháng gắn bó", MonthlyCharges: "Tiền mỗi tháng", SeniorCitizen: "Người cao tuổi", Contract: "Hợp đồng",
  InternetService: "Internet", PaymentMethod: "Thanh toán", OnlineSecurity: "Bảo mật", TechSupport: "Hỗ trợ kỹ thuật",
};
const VALUES = {
  "Month-to-month": "Theo tháng", "One year": "1 năm", "Two year": "2 năm", "Fiber optic": "Cáp quang", DSL: "DSL", No: "Không", Yes: "Có",
  "No internet service": "Không có Internet", "Bank transfer (automatic)": "Chuyển khoản tự động", "Credit card (automatic)": "Thẻ tín dụng tự động",
  "Electronic check": "Séc điện tử", "Mailed check": "Séc qua thư",
};
const MODELS = {
  lr: "Logistic Regression", lr_smote: "Logistic Regression + SMOTE", xgb_depth2: "XGBoost (sâu 2)", xgb_depth3: "XGBoost (sâu 3)",
  xgb_depth4: "XGBoost (sâu 4)", xgb_weighted: "XGBoost + trọng số lớp", xgb_smote: "XGBoost + SMOTE",
};
const PRESETS = {
  high: { tenure: 2, MonthlyCharges: 95, SeniorCitizen: 0, Contract: "Month-to-month", InternetService: "Fiber optic", PaymentMethod: "Electronic check", OnlineSecurity: "No", TechSupport: "No" },
  low: { tenure: 60, MonthlyCharges: 50, SeniorCitizen: 0, Contract: "Two year", InternetService: "DSL", PaymentMethod: "Credit card (automatic)", OnlineSecurity: "Yes", TechSupport: "Yes" },
};
const NO_NET = "No internet service";
const CATEGORICAL = ["Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport"];
const ADDONS = ["OnlineSecurity", "TechSupport"];
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const pct = (p) => (p * 100).toFixed(1) + "%";
const charts = {};
let requestId = 0;

Chart.defaults.color = COLORS.ink2;
Chart.defaults.borderColor = COLORS.grid;
Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
Chart.defaults.maintainAspectRatio = false;
if (reduceMotion) Chart.defaults.animation = false;

function layout(extra) {
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", margin: { l: 130, r: 24, t: 8, b: 40 },
    font: { family: Chart.defaults.font.family, color: COLORS.ink2, size: 12 }, hoverlabel: { bgcolor: "#1a2032", font: { color: COLORS.ink } }, ...extra,
  };
}
const PLOT = { displayModeBar: false, responsive: true };

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function icon(name) {
  const node = el("i", undefined, "fa-solid " + name);
  node.setAttribute("aria-hidden", "true");
  return node;
}

async function api(path, body) {
  let response;
  try {
    response = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  } catch {
    throw Object.assign(new Error("Không kết nối được máy chủ."), { fields: [] });
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = (data && data.detail) || [];
    const fields = detail.map((d) => d.loc[d.loc.length - 1]).filter((f) => f in FEATURES);
    const names = [...new Set(fields.map((f) => FEATURES[f]))];
    throw Object.assign(new Error(names.length ? "Giá trị chưa hợp lệ: " + names.join(", ") + "." : "Yêu cầu chưa hợp lệ."), { fields });
  }
  return data;
}

function fillSelect(id, values) {
  $(id).replaceChildren(...values.map((v) => {
    const option = el("option", VALUES[v] || v);
    option.value = v;
    return option;
  }));
}

function syncInternet() {
  const none = $("InternetService").value === "No";
  for (const id of ADDONS) {
    const select = $(id);
    for (const option of select.options) option.hidden = option.disabled = (option.value === NO_NET) !== none;
    if ((select.value === NO_NET) !== none || select.selectedIndex < 0) select.value = none ? NO_NET : "No";
    select.disabled = none;
  }
}

function setForm(values) {
  for (const [key, value] of Object.entries(values)) $(key).value = String(value);
  syncInternet();
  $("tenure-out").textContent = $("tenure").value + " tháng";
}

function readForm() {
  const customer = {
    tenure: Number($("tenure").value),
    MonthlyCharges: $("MonthlyCharges").value === "" ? null : Number($("MonthlyCharges").value),
    SeniorCitizen: Number($("SeniorCitizen").value),
  };
  for (const id of CATEGORICAL) customer[id] = $(id).value;
  return { model: $("model").value, customer };
}

function drawGauge(p, high) {
  const color = p === null ? COLORS.track : high ? COLORS.red : COLORS.blue;
  const data = [p === null ? 0 : p, p === null ? 1 : 1 - p];
  if (!charts.gauge) {
    charts.gauge = new Chart($("gauge"), {
      type: "doughnut",
      data: { labels: ["Rời mạng", "Ở lại"], datasets: [{ data, backgroundColor: [color, COLORS.track], borderWidth: 0, borderRadius: 4 }] },
      options: { cutout: "80%", plugins: { legend: { display: false }, tooltip: { enabled: false } }, events: [] },
    });
    return;
  }
  charts.gauge.data.datasets[0].data = data;
  charts.gauge.data.datasets[0].backgroundColor = [color, COLORS.track];
  charts.gauge.update();
}

function drawContributions(explanation) {
  const items = [...explanation.contributions].reverse();
  const max = Math.max(...items.map((c) => Math.abs(c.log_odds)), 0.1) * 1.3;
  const name = (c) => FEATURES[c.feature];
  const value = (c) => (c.feature === "SeniorCitizen" ? (c.value ? "Có" : "Không") : VALUES[c.value] || String(c.value));
  Plotly.react("contrib", [{
    type: "bar", orientation: "h", y: items.map(name), x: items.map((c) => c.log_odds), width: 0.55,
    marker: { color: items.map((c) => (c.log_odds >= 0 ? COLORS.red : COLORS.blue)), cornerradius: 4 },
    text: items.map((c) => (c.log_odds >= 0 ? "+" : "−") + Math.abs(c.log_odds).toFixed(2)), textposition: "outside", textfont: { color: COLORS.ink2 }, cliponaxis: false,
    customdata: items.map(value), hovertemplate: "%{y}: %{customdata}<br>%{x:+.2f} log-odds<extra></extra>",
  }], layout({
    xaxis: { range: [-max, max], zeroline: true, zerolinecolor: "rgba(255,255,255,0.35)", gridcolor: COLORS.grid, title: { text: "← giảm rủi ro      tăng rủi ro →", font: { size: 12 } }, fixedrange: true },
    yaxis: { fixedrange: true, automargin: true, ticksuffix: "  " }, showlegend: false,
  }), PLOT);
}

function drawCurve(curve, current) {
  const datasets = [
    { label: "Xác suất rời mạng", data: curve.probability.map((p) => p * 100), borderColor: COLORS.blue, backgroundColor: "rgba(57,135,229,0.12)", fill: true, borderWidth: 2, pointRadius: 0, pointHitRadius: 14, tension: 0 },
    { label: "Ngưỡng", data: curve.tenure.map(() => curve.threshold * 100), borderColor: COLORS.ink3, borderWidth: 1.5, pointRadius: 0, pointHitRadius: 0 },
  ];
  const nearest = curve.tenure.reduce((best, m, i) => (Math.abs(m - current) < Math.abs(curve.tenure[best] - current) ? i : best), 0);
  datasets[0].pointRadius = curve.tenure.map((_, i) => (i === nearest ? 5 : 0));
  datasets[0].pointBackgroundColor = COLORS.blue;
  datasets[0].pointBorderColor = COLORS.ink;
  if (charts.curve) {
    charts.curve.data = { labels: curve.tenure, datasets };
    charts.curve.update();
    return;
  }
  charts.curve = new Chart($("curve"), {
    type: "line", data: { labels: curve.tenure, datasets },
    options: {
      interaction: { mode: "index", intersect: false },
      scales: { y: { min: 0, max: 100, ticks: { callback: (v) => v + "%", stepSize: 25 } }, x: { title: { display: true, text: "Số tháng gắn bó" }, grid: { display: false } } },
      plugins: { legend: { labels: { usePointStyle: true, pointStyle: "line" } }, tooltip: { callbacks: { title: (i) => i[0].label + " tháng", label: (i) => `${i.dataset.label}: ${i.parsed.y.toFixed(1)}%` } } },
    },
  });
}

function showPrediction(result) {
  const high = result.will_churn;
  drawGauge(result.churn_probability, high);
  $("prob").textContent = pct(result.churn_probability);
  const label = $("label");
  label.className = "status " + (high ? "high" : "low");
  label.replaceChildren(icon(high ? "fa-triangle-exclamation" : "fa-circle-check"), el("span", high ? "Có nguy cơ rời mạng" : "Nhiều khả năng ở lại"));
  $("detail").textContent = `${result.model_label}, ngưỡng ${pct(result.threshold)}. Điểm xuất phát: ${pct(result.explanation.base_probability)}.`;
  $("warnings").replaceChildren(...result.warnings.map((w) => {
    const item = el("li");
    item.append(icon("fa-circle-exclamation"), " " + w);
    return item;
  }));
  drawContributions(result.explanation);
}

async function submit(event) {
  if (event) event.preventDefault();
  const id = ++requestId;
  const body = readForm();
  $("submit").disabled = true;
  $("form-error").hidden = true;
  for (const key of Object.keys(FEATURES)) $(key).removeAttribute("aria-invalid");
  try {
    const [result, curve] = await Promise.all([api("/api/predict", body), api("/api/tenure-curve", body)]);
    if (id !== requestId) return;
    document.querySelector(".results").classList.remove("stale");
    showPrediction(result);
    drawCurve(curve, body.customer.tenure);
  } catch (err) {
    if (id !== requestId) return;
    document.querySelector(".results").classList.add("stale");
    $("form-error").textContent = err.message;
    $("form-error").hidden = false;
    for (const key of err.fields || []) $(key).setAttribute("aria-invalid", "true");
  } finally {
    if (id === requestId) $("submit").disabled = false;
  }
}

function kpi(label, value, note) {
  const node = el("div", undefined, "kpi glass");
  node.append(el("span", label), el("strong", value), el("small", note));
  return node;
}

function drawInsights(d) {
  const best = d.metrics.xgb;
  $("kpis").replaceChildren(
    kpi("Khách hàng", d.customers.toLocaleString("vi-VN"), `churn ${d.churn_rate}%`),
    kpi("ROC-AUC (test)", best.roc_auc.toFixed(3), `Logistic: ${d.metrics.lr.roc_auc.toFixed(3)}`),
    kpi("PR-AUC (test)", best.pr_auc.toFixed(3), `Logistic: ${d.metrics.lr.pr_auc.toFixed(3)}`),
    kpi("Recall (test)", pct(best.recall), `precision ${pct(best.precision)}`),
  );

  const h = d.heatmap;
  Plotly.react("heatmap", [{
    type: "heatmap", z: h.rate, x: h.cols.map((c) => VALUES[c]), y: h.rows.map((r) => (r === "No" ? "Không Internet" : VALUES[r])),
    zmin: 0, zmax: 60, xgap: 3, ygap: 3, showscale: false, customdata: h.n,
    colorscale: [[0, "#12284a"], [0.5, "#256abf"], [1, "#9ec5f4"]],
    hovertemplate: "%{y}, %{x}<br>churn %{z}%<br>%{customdata} khách<extra></extra>",
  }], layout({
    margin: { l: 110, r: 12, t: 8, b: 36 },
    xaxis: { fixedrange: true, showgrid: false }, yaxis: { fixedrange: true, showgrid: false, ticksuffix: "  " },
    annotations: h.rate.flatMap((row, i) => row.map((v, j) => ({ x: j, y: i, xref: "x", yref: "y", text: `<b>${v}%</b>`, showarrow: false, font: { color: v > 45 ? "#0b0b0b" : "#ffffff", size: 14 } }))),
  }), PLOT);

  charts.tenure = new Chart($("tenure-chart"), {
    type: "bar", data: { labels: d.tenure.labels, datasets: [{ label: "Tỷ lệ churn", data: d.tenure.rate, backgroundColor: COLORS.blue, borderRadius: 4, borderSkipped: "bottom", maxBarThickness: 24 }] },
    options: {
      scales: { y: { beginAtZero: true, ticks: { callback: (v) => v + "%" } }, x: { grid: { display: false }, title: { display: true, text: "Số tháng gắn bó" } } },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (i) => `${i.parsed.y}% (${d.tenure.n[i.dataIndex]} khách)` } } },
    },
  });

  const g = d.gains;
  const line = (name, y, color, width) => ({ type: "scatter", mode: "lines", name, x: g.contacted, y, line: { color, width }, hovertemplate: `${name}: %{y}%<extra></extra>` });
  Plotly.react("gains", [
    line("XGBoost", g.xgb, COLORS.blue, 2), line("Logistic", g.lr, COLORS.orange, 2), line("Gọi ngẫu nhiên", g.contacted, COLORS.ink3, 1.5),
  ], layout({
    margin: { l: 52, r: 26, t: 30, b: 46 }, hovermode: "x unified", legend: { orientation: "h", y: 1.14, x: 0 },
    xaxis: { title: { text: "% khách được liên hệ" }, gridcolor: COLORS.grid, fixedrange: true, range: [0, 100.5], zeroline: false, ticksuffix: "%", tickvals: [0, 20, 40, 60, 80, 100] },
    yaxis: { gridcolor: COLORS.grid, fixedrange: true, range: [0, 102], zeroline: false, ticksuffix: "%" },
  }), PLOT);

  charts.importance = new Chart($("importance"), {
    type: "bar", data: { labels: d.importance.features.map((f) => FEATURES[f]), datasets: [{ label: "|SHAP| trung bình", data: d.importance.values, backgroundColor: COLORS.blue, borderRadius: 4, borderSkipped: "left", maxBarThickness: 18 }] },
    options: { indexAxis: "y", scales: { x: { beginAtZero: true }, y: { grid: { display: false } } }, plugins: { legend: { display: false } } },
  });

  $("comparison").tBodies[0].replaceChildren(...d.comparison.map((r) => {
    const row = el("tr", undefined, r.name === d.chosen ? "chosen" : "");
    row.append(el("td", (MODELS[r.name] || r.name) + (r.name === d.chosen ? " (được chọn)" : "")), el("td", r.pr_auc.toFixed(4)), el("td", r.roc_auc.toFixed(4)), el("td", r.brier.toFixed(4)));
    return row;
  }));
}

function showOnline() {
  $("offline").hidden = navigator.onLine;
}

async function start() {
  drawGauge(null, false);
  showOnline();
  addEventListener("online", showOnline);
  addEventListener("offline", showOnline);

  const [options, models, insights] = await Promise.all([api("/api/options"), api("/api/models"), api("/api/insights")]);
  for (const id of CATEGORICAL) fillSelect(id, options.categories[id]);
  [$("tenure").min, $("tenure").max] = options.ranges.tenure;
  $("model").replaceChildren(...models.models.map((m) => {
    const option = el("option", m.label + (m.name === models.champion ? " (khuyên dùng)" : ""));
    option.value = m.name;
    return option;
  }));
  $("model").value = models.champion;
  setForm(PRESETS.high);
  drawInsights(insights);

  $("form").addEventListener("submit", submit);
  $("InternetService").addEventListener("change", syncInternet);
  $("tenure").addEventListener("input", () => { $("tenure-out").textContent = $("tenure").value + " tháng"; });
  for (const chip of document.querySelectorAll("[data-preset]")) {
    chip.addEventListener("click", () => { setForm(PRESETS[chip.dataset.preset]); submit(); });
  }
  await submit();
}

let installPrompt = null;
addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  installPrompt = event;
  $("install").hidden = false;
});
addEventListener("appinstalled", () => { $("install").hidden = true; });
$("install").addEventListener("click", async () => {
  if (!installPrompt) return;
  installPrompt.prompt();
  await installPrompt.userChoice;
  installPrompt = null;
  $("install").hidden = true;
});

if ("serviceWorker" in navigator) {
  addEventListener("load", () => navigator.serviceWorker.register("/sw.js").catch(() => {}));
}

start().catch((err) => {
  $("form-error").textContent = err.message;
  $("form-error").hidden = false;
});
