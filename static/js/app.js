/**
 * AQI Machine Learning System - Client Application Logic
 * Integrates Chart.js for data science visualization, handles live dual/triple-model inference,
 * and manages Dark / Light mode theme toggling.
 */

// Global Chart references for dynamic redraw
let metricChart = null;
let fiChart = null;
let timeseriesChart = null;
let profileChart = null;

// Cached metrics & timeseries data for theme re-rendering
let cachedMetricsData = null;

// Presets data mapping
const PRESETS = {
  clean: { 'PM2.5': 15.0, 'PM10': 32.0, 'NO2': 12.0, 'CO': 0.5, 'O3': 18.0 },
  metro: { 'PM2.5': 60.0, 'PM10': 125.0, 'NO2': 42.0, 'CO': 1.4, 'O3': 45.0 },
  inversion: { 'PM2.5': 155.0, 'PM10': 250.0, 'NO2': 75.0, 'CO': 2.4, 'O3': 65.0 },
  smog: { 'PM2.5': 380.0, 'PM10': 520.0, 'NO2': 115.0, 'CO': 4.1, 'O3': 92.0 }
};

document.addEventListener("DOMContentLoaded", () => {
  initThemeToggle();
  initTabs();
  initSliders();
  initPresets();
  initRetrainButton();
  loadEvaluationMetrics();
  triggerPrediction(); // Initial simulation call
});

/* --------------------------------------------------------------------------
   1. Theme Management (Light / Dark Mode)
   -------------------------------------------------------------------------- */
function initThemeToggle() {
  const toggleBtn = document.getElementById("btn-theme-toggle");
  const themeIcon = document.getElementById("theme-icon");
  const themeLabel = document.getElementById("theme-label");

  // Load saved theme or default to dark
  const savedTheme = localStorage.getItem("aqi_theme") || "dark";
  applyTheme(savedTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
      const newTheme = currentTheme === "dark" ? "light" : "dark";
      applyTheme(newTheme);
      localStorage.setItem("aqi_theme", newTheme);

      // Re-render charts with updated theme colors
      if (cachedMetricsData) {
        renderMetricComparisonChart(cachedMetricsData.metrics);
        renderFeatureImportanceChart(cachedMetricsData.feature_importances);
        renderTimeseriesChart(cachedMetricsData.timeseries);
      }
      triggerPrediction();
    });
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    if (theme === "light") {
      if (themeIcon) themeIcon.textContent = "🌙";
      if (themeLabel) themeLabel.textContent = "Dark";
    } else {
      if (themeIcon) themeIcon.textContent = "☀️";
      if (themeLabel) themeLabel.textContent = "Light";
    }
  }
}

function getThemeColors() {
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  return {
    isLight,
    text: isLight ? "#475569" : "#a1a1aa",
    dimText: isLight ? "#64748b" : "#71717a",
    mainText: isLight ? "#0f172a" : "#f4f4f5",
    grid: isLight ? "rgba(0, 0, 0, 0.06)" : "rgba(255, 255, 255, 0.05)",
    gridSubtle: isLight ? "rgba(0, 0, 0, 0.03)" : "rgba(255, 255, 255, 0.03)",
    tooltipBg: isLight ? "#ffffff" : "#18181b",
    tooltipBorder: isLight ? "rgba(0,0,0,0.12)" : "rgba(255,255,255,0.15)",
    tooltipTitle: isLight ? "#0f172a" : "#ffffff",
    tooltipBody: isLight ? "#334155" : "#f4f4f5"
  };
}

/* --------------------------------------------------------------------------
   2. Tab Navigation (2 Core Tabs)
   -------------------------------------------------------------------------- */
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");

      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const activeContent = document.getElementById(targetId);
      if (activeContent) {
        activeContent.classList.add("active");
      }
    });
  });
}

/* --------------------------------------------------------------------------
   3. Fetch Evaluation Metrics & Render Charts
   -------------------------------------------------------------------------- */
async function loadEvaluationMetrics() {
  try {
    const res = await fetch("/api/metrics");
    const data = await res.json();

    if (data.status !== "success") {
      console.error("Failed to load metrics:", data.message);
      return;
    }

    cachedMetricsData = data;
    const { metrics, feature_importances, diagnostics, timeseries } = data;

    // Populate diagnostics in top header status
    const statusText = document.getElementById("dataset-status-text");
    if (statusText && diagnostics) {
      statusText.textContent = `CPCB Archive | ${diagnostics.total_records.toLocaleString()} Records (${diagnostics.date_min} to ${diagnostics.date_max})`;
    }

    // Populate KPI Cards
    populateKPICards(metrics);

    // Render Tab 1 Charts
    renderMetricComparisonChart(metrics);
    renderFeatureImportanceChart(feature_importances);
    renderTimeseriesChart(timeseries);

  } catch (err) {
    console.error("Error fetching metrics:", err);
  }
}

function populateKPICards(metrics) {
  const rf = metrics.Random_Forest;
  const svr = metrics.Support_Vector_Regression;
  const gbr = metrics.Gradient_Boosting;

  // RMSE
  document.getElementById("kpi-rf-rmse").textContent = rf.rmse.toFixed(2);
  document.getElementById("kpi-svr-rmse").textContent = svr.rmse.toFixed(2);
  document.getElementById("kpi-gbr-rmse").textContent = gbr.rmse.toFixed(2);

  let bestRmseModel = "SVR";
  let minRmse = svr.rmse;
  if (rf.rmse < minRmse) { minRmse = rf.rmse; bestRmseModel = "Random Forest"; }
  if (gbr.rmse < minRmse) { minRmse = gbr.rmse; bestRmseModel = "Gradient Boosting"; }
  document.getElementById("kpi-rmse-winner").textContent = `Best: ${bestRmseModel}`;

  // MAE
  document.getElementById("kpi-rf-mae").textContent = rf.mae.toFixed(2);
  document.getElementById("kpi-svr-mae").textContent = svr.mae.toFixed(2);
  document.getElementById("kpi-gbr-mae").textContent = gbr.mae.toFixed(2);

  let bestMaeModel = "SVR";
  let minMae = svr.mae;
  if (rf.mae < minMae) { minMae = rf.mae; bestMaeModel = "Random Forest"; }
  if (gbr.mae < minMae) { minMae = gbr.mae; bestMaeModel = "Gradient Boosting"; }
  document.getElementById("kpi-mae-winner").textContent = `Best: ${bestMaeModel}`;

  // R²
  document.getElementById("kpi-rf-r2").textContent = rf.r2_score.toFixed(4);
  document.getElementById("kpi-svr-r2").textContent = svr.r2_score.toFixed(4);
  document.getElementById("kpi-gbr-r2").textContent = gbr.r2_score.toFixed(4);

  let bestR2Model = "SVR";
  let maxR2 = svr.r2_score;
  if (rf.r2_score > maxR2) { maxR2 = rf.r2_score; bestR2Model = "Random Forest"; }
  if (gbr.r2_score > maxR2) { maxR2 = gbr.r2_score; bestR2Model = "Gradient Boosting"; }
  document.getElementById("kpi-r2-winner").textContent = `Best: ${bestR2Model}`;
}

/* --------------------------------------------------------------------------
   Chart 1: Metric Comparison (Side-by-Side Grouped Bar)
   -------------------------------------------------------------------------- */
function renderMetricComparisonChart(metrics) {
  const ctx = document.getElementById("chart-metrics").getContext("2d");
  if (metricChart) metricChart.destroy();

  const tc = getThemeColors();
  const rf = metrics.Random_Forest;
  const svr = metrics.Support_Vector_Regression;
  const gbr = metrics.Gradient_Boosting;

  metricChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["RMSE (Lower Better)", "MAE (Lower Better)", "R² Score (x100)"],
      datasets: [
        {
          label: "Random Forest",
          data: [rf.rmse, rf.mae, rf.r2_score * 100],
          backgroundColor: "#3b82f6",
          borderRadius: 4
        },
        {
          label: "SVR",
          data: [svr.rmse, svr.mae, svr.r2_score * 100],
          backgroundColor: "#f59e0b",
          borderRadius: 4
        },
        {
          label: "Gradient Boosting",
          data: [gbr.rmse, gbr.mae, gbr.r2_score * 100],
          backgroundColor: "#8b5cf6",
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: tc.text, font: { family: "Inter", size: 12 } }
        },
        tooltip: {
          backgroundColor: tc.tooltipBg,
          borderColor: tc.tooltipBorder,
          borderWidth: 1,
          titleColor: tc.tooltipTitle,
          bodyColor: tc.tooltipBody
        }
      },
      scales: {
        x: {
          ticks: { color: tc.text, font: { family: "Inter" } },
          grid: { color: tc.grid }
        },
        y: {
          ticks: { color: tc.dimText, font: { family: "Inter" } },
          grid: { color: tc.grid }
        }
      }
    }
  });
}

/* --------------------------------------------------------------------------
   Chart 2: Feature Importance (RF vs. Gradient Boosting)
   -------------------------------------------------------------------------- */
function renderFeatureImportanceChart(fiData) {
  const ctx = document.getElementById("chart-feature-importance").getContext("2d");
  if (fiChart) fiChart.destroy();

  const tc = getThemeColors();
  const rfFi = fiData.Random_Forest || {};
  const gbrFi = fiData.Gradient_Boosting || {};

  const labels = ["PM2.5", "PM10", "NO2", "CO", "O3"];
  const rfVals = labels.map(l => (rfFi[l] || 0) * 100);
  const gbrVals = labels.map(l => (gbrFi[l] || 0) * 100);

  fiChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Random Forest (MDI)",
          data: rfVals,
          backgroundColor: "rgba(59, 130, 246, 0.85)",
          borderRadius: 4
        },
        {
          label: "Gradient Boosting (Impurity)",
          data: gbrVals,
          backgroundColor: "rgba(139, 92, 246, 0.85)",
          borderRadius: 4
        }
      ]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: tc.text, font: { family: "Inter", size: 12 } }
        },
        tooltip: {
          backgroundColor: tc.tooltipBg,
          borderColor: tc.tooltipBorder,
          borderWidth: 1,
          titleColor: tc.tooltipTitle,
          bodyColor: tc.tooltipBody,
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(1)}%`
          }
        }
      },
      scales: {
        x: {
          ticks: {
            color: tc.dimText,
            callback: (v) => `${v}%`
          },
          grid: { color: tc.grid }
        },
        y: {
          ticks: { color: tc.mainText, font: { weight: "600" } },
          grid: { display: false }
        }
      }
    }
  });
}

/* --------------------------------------------------------------------------
   Chart 3: Chronological Time-Series Horizon (Out-of-Sample Test)
   -------------------------------------------------------------------------- */
function renderTimeseriesChart(tsData) {
  const ctx = document.getElementById("chart-timeseries").getContext("2d");
  if (timeseriesChart) timeseriesChart.destroy();

  const tc = getThemeColors();

  timeseriesChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: tsData.dates,
      datasets: [
        {
          label: "Actual Recorded AQI",
          data: tsData.actual,
          borderColor: tc.isLight ? "#64748b" : "#94a3b8",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.15
        },
        {
          label: "Random Forest",
          data: tsData.rf,
          borderColor: "#3b82f6",
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.15
        },
        {
          label: "SVR",
          data: tsData.svr,
          borderColor: "#f59e0b",
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0,
          tension: 0.15
        },
        {
          label: "Gradient Boosting",
          data: tsData.gbr,
          borderColor: "#8b5cf6",
          borderWidth: 1.5,
          borderDash: [2, 2],
          pointRadius: 0,
          tension: 0.15
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: {
          labels: { color: tc.text, font: { family: "Inter", size: 12 } }
        },
        tooltip: {
          backgroundColor: tc.tooltipBg,
          borderColor: tc.tooltipBorder,
          borderWidth: 1,
          titleColor: tc.tooltipTitle,
          bodyColor: tc.tooltipBody
        }
      },
      scales: {
        x: {
          ticks: { color: tc.dimText, maxTicksLimit: 12 },
          grid: { color: tc.gridSubtle }
        },
        y: {
          title: { display: true, text: "Air Quality Index (AQI)", color: tc.text },
          ticks: { color: tc.dimText },
          grid: { color: tc.grid }
        }
      }
    }
  });
}

/* --------------------------------------------------------------------------
   4. Tab 2 Simulator Logic & Dynamic Inference
   -------------------------------------------------------------------------- */
function initSliders() {
  const sliders = ["pm25", "pm10", "no2", "co", "o3"];
  sliders.forEach(key => {
    const slider = document.getElementById(`slider-${key}`);
    const badge = document.getElementById(`val-${key}`);

    slider.addEventListener("input", () => {
      badge.textContent = parseFloat(slider.value).toFixed(key === "co" ? 1 : 0);
      triggerPrediction();
    });
  });
}

function initPresets() {
  const presetBtns = document.querySelectorAll(".btn-preset");
  presetBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      presetBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      const presetKey = btn.getAttribute("data-preset");
      const values = PRESETS[presetKey];
      if (values) {
        document.getElementById("slider-pm25").value = values['PM2.5'];
        document.getElementById("val-pm25").textContent = values['PM2.5'].toFixed(0);

        document.getElementById("slider-pm10").value = values['PM10'];
        document.getElementById("val-pm10").textContent = values['PM10'].toFixed(0);

        document.getElementById("slider-no2").value = values['NO2'];
        document.getElementById("val-no2").textContent = values['NO2'].toFixed(0);

        document.getElementById("slider-co").value = values['CO'];
        document.getElementById("val-co").textContent = values['CO'].toFixed(1);

        document.getElementById("slider-o3").value = values['O3'];
        document.getElementById("val-o3").textContent = values['O3'].toFixed(0);

        triggerPrediction();
      }
    });
  });
}

let debounceTimer = null;
function triggerPrediction() {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(executePredictionRequest, 60);
}

async function executePredictionRequest() {
  const payload = {
    'PM2.5': parseFloat(document.getElementById("slider-pm25").value),
    'PM10': parseFloat(document.getElementById("slider-pm10").value),
    'NO2': parseFloat(document.getElementById("slider-no2").value),
    'CO': parseFloat(document.getElementById("slider-co").value),
    'O3': parseFloat(document.getElementById("slider-o3").value)
  };

  // Update real-time profile chart
  updateProfileChart(payload);

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (data.status === "success") {
      updatePredictionUI(data);
    }
  } catch (err) {
    console.error("Prediction failed:", err);
  }
}

function updatePredictionUI(data) {
  const { predictions, cpcb } = data;

  // Individual model forecast cards
  document.getElementById("pred-val-rf").textContent = predictions.rf.toFixed(1);
  document.getElementById("pred-val-svr").textContent = predictions.svr.toFixed(1);
  document.getElementById("pred-val-gbr").textContent = predictions.gbr.toFixed(1);

  // Hero consensus card
  const heroConsensus = document.getElementById("hero-consensus-val");
  heroConsensus.textContent = predictions.consensus.toFixed(1);

  const pill = document.getElementById("hero-aqi-pill");
  pill.textContent = `${cpcb.category} (${cpcb.range})`;
  pill.style.backgroundColor = cpcb.color;
  pill.style.color = (cpcb.category === "Moderate" || cpcb.category === "Satisfactory") ? "#0f172a" : "#ffffff";

  const advisory = document.getElementById("hero-advisory-text");
  advisory.textContent = cpcb.advisory;
}

/* --------------------------------------------------------------------------
   Chart 4: Input Pollutant Profile Chart
   -------------------------------------------------------------------------- */
function updateProfileChart(inputs) {
  const ctx = document.getElementById("chart-profile").getContext("2d");
  const tc = getThemeColors();

  const labels = ["PM2.5", "PM10", "NO2", "CO (x10)", "O3"];
  const values = [inputs['PM2.5'], inputs['PM10'], inputs['NO2'], inputs['CO'] * 10, inputs['O3']];

  if (!profileChart) {
    profileChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          data: values,
          backgroundColor: ["#3b82f6", "#60a5fa", "#93c5fd", "#f59e0b", "#fbbf24"],
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: tc.tooltipBg,
            borderColor: tc.tooltipBorder,
            borderWidth: 1,
            titleColor: tc.tooltipTitle,
            bodyColor: tc.tooltipBody,
            callbacks: {
              label: (ctx) => `Concentration: ${ctx.raw.toFixed(1)}`
            }
          }
        },
        scales: {
          x: { ticks: { color: tc.text, font: { size: 11 } }, grid: { display: false } },
          y: { ticks: { color: tc.dimText, font: { size: 10 } }, grid: { color: tc.grid } }
        }
      }
    });
  } else {
    profileChart.data.datasets[0].data = values;
    profileChart.options.scales.x.ticks.color = tc.text;
    profileChart.options.scales.y.ticks.color = tc.dimText;
    profileChart.options.scales.y.grid.color = tc.grid;
    profileChart.options.plugins.tooltip.backgroundColor = tc.tooltipBg;
    profileChart.options.plugins.tooltip.borderColor = tc.tooltipBorder;
    profileChart.options.plugins.tooltip.titleColor = tc.tooltipTitle;
    profileChart.options.plugins.tooltip.bodyColor = tc.tooltipBody;
    profileChart.update("none");
  }
}

/* --------------------------------------------------------------------------
   5. Retraining Trigger
   -------------------------------------------------------------------------- */
function initRetrainButton() {
  const retrainBtn = document.getElementById("btn-retrain");
  if (!retrainBtn) return;

  retrainBtn.addEventListener("click", async () => {
    retrainBtn.disabled = true;
    const originalText = retrainBtn.innerHTML;
    retrainBtn.innerHTML = `<span>⏳ Retraining Models...</span>`;

    try {
      const res = await fetch("/api/retrain", { method: "POST" });
      const data = await res.json();
      if (data.status === "success") {
        alert("✅ All 3 ML Models successfully retrained!");
        await loadEvaluationMetrics();
        triggerPrediction();
      } else {
        alert("Retraining failed: " + data.message);
      }
    } catch (err) {
      alert("Retraining request failed: " + err);
    } finally {
      retrainBtn.disabled = false;
      retrainBtn.innerHTML = originalText;
    }
  });
}
