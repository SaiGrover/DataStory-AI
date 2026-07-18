import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";
import type { EdaResponse } from "./api";
import type { DatasetResponse, TrainResult } from "./types";

const COLORS = {
  ink: [29, 38, 31] as [number, number, number],
  sage: [91, 125, 77] as [number, number, number],
  sageDark: [61, 91, 52] as [number, number, number],
  sagePale: [238, 245, 233] as [number, number, number],
  line: [218, 226, 214] as [number, number, number],
  muted: [101, 112, 103] as [number, number, number],
  white: [255, 255, 255] as [number, number, number],
};

type PdfExportOptions = {
  filename: string;
  title: string;
  subtitle: string;
  datasetName: string;
  markdown: string;
  metrics?: Array<{ label: string; value: string }>;
  charts?: PdfChart[];
};

export type PdfChart =
  | { type: "bar"; title: string; subtitle?: string; labels: string[]; values: number[]; valueSuffix?: string; horizontal?: boolean }
  | { type: "grouped"; title: string; subtitle?: string; labels: string[]; series: Array<{ name: string; values: number[] }> }
  | { type: "donut"; title: string; subtitle?: string; labels: string[]; values: number[] }
  | { type: "heatmap"; title: string; subtitle?: string; labels: string[]; values: number[][] };

export type EdaPdfData = {
  dataset: DatasetResponse;
  eda?: EdaResponse;
  findings: string[];
  recommendedTarget?: string;
};

export type ModelingPdfData = {
  dataset: DatasetResponse;
  target: string;
  taskType: string;
  testSize: number;
  cvFolds: number;
  imbalanceStrategy?: string;
  results: TrainResult[];
};

export function downloadEdaReportPdf({ dataset, eda, findings, recommendedTarget }: EdaPdfData) {
  const profile = dataset.profile;
  const details = profile.column_details ?? [];
  const missing = details
    .map((row) => ({ name: String(row.Column ?? "-"), pct: Number(row["Null %"] ?? 0) }))
    .filter((row) => row.pct > 0)
    .sort((a, b) => b.pct - a.pct)
    .slice(0, 10);
  const numeric = eda?.numeric_profiles?.slice(0, 10) ?? [];
  const categorical = eda?.categorical_profiles?.slice(0, 8) ?? [];
  const correlations = strongestCorrelations(eda?.correlation);
  const base = safeBaseName(dataset.filename);
  const charts = buildEdaCharts(dataset, eda, missing);

  const markdown = [
    "## Executive Summary",
    `DataStory AI profiled **${dataset.filename}** to assess structure, quality, distributions, and relationships. The dataset contains **${dataset.rows.toLocaleString()} rows** and **${dataset.columns.toLocaleString()} columns**, with a data health score of **${profile.health_score ?? 0}/100**.`,
    `The recommended target candidate is **${recommendedTarget || "not selected"}**. Confirm that it matches the real prediction or decision objective before modeling.`,
    "",
    "## Dataset Snapshot",
    "| Metric | Value |",
    "| --- | ---: |",
    `| Rows | ${dataset.rows.toLocaleString()} |`,
    `| Columns | ${dataset.columns.toLocaleString()} |`,
    `| Numeric columns | ${formatWhole(profile.numeric_cols)} |`,
    `| Categorical columns | ${formatWhole(profile.categorical_cols)} |`,
    `| Date columns | ${formatWhole(profile.date_cols)} |`,
    `| Missing cells | ${formatWhole(profile.total_missing)} |`,
    `| Duplicate rows | ${formatWhole(profile.duplicates)} |`,
    `| Data health | ${profile.health_score ?? 0}/100 |`,
    "",
    "## Key Findings",
    ...findings.map((item) => `- ${item}`),
    ...(profile.warnings ?? []).map((item) => `- ${item.title}: ${item.message}`),
    "",
    "## Column Profile",
    "| Column | Type | Missing % | Unique | Top value |",
    "| --- | --- | ---: | ---: | --- |",
    ...details.slice(0, 24).map((row) => `| ${md(row.Column)} | ${md(row.Type)} | ${formatNumber(row["Null %"])} | ${md(row.Unique)} | ${md(String(row["Top Value"] ?? "-").slice(0, 28))} |`),
    details.length > 24 ? `| ... | ${details.length - 24} additional columns |  |  |  |` : "",
    "",
    "## Missing Values",
    missing.length ? "| Column | Missing % |\n| --- | ---: |\n" + missing.map((row) => `| ${md(row.name)} | ${formatNumber(row.pct)}% |`).join("\n") : "No missing values were detected in the current dataset profile.",
    "",
    "## Numeric Distributions",
    numeric.length ? "| Feature | Min | Q1 | Median | Q3 | Max | Outliers |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n" + numeric.map((item) => `| ${md(item.column)} | ${formatNumber(item.box.min)} | ${formatNumber(item.box.q1)} | ${formatNumber(item.box.median)} | ${formatNumber(item.box.q3)} | ${formatNumber(item.box.max)} | ${item.box.outliers.toLocaleString()} |`).join("\n") : "No numeric distribution profiles are available.",
    "",
    "## Categorical Highlights",
    ...(categorical.length ? categorical.map((item) => `- ${item.column}: ${item.labels.slice(0, 5).map((label, index) => `${label} (${item.counts[index]?.toLocaleString() ?? 0})`).join(", ")}`) : ["No categorical profiles are available."]),
    "",
    "## Strongest Correlations",
    correlations.length ? "| Feature pair | Correlation |\n| --- | ---: |\n" + correlations.map((row) => `| ${md(row.pair)} | ${formatNumber(row.value)} |`).join("\n") : "No numeric correlation matrix is available.",
    "",
    "## Recommended Next Steps",
    "- Validate the suggested target against the business or research question.",
    "- Resolve missing values, duplicates, and suspicious outliers before final modeling.",
    "- Review high-cardinality and ID-like columns before feature selection.",
    "- Re-export this report after each material cleaning step to preserve an audit trail.",
    "",
    "## Technical Notes",
    "This report is generated from the active uploaded dataset and the latest DataStory EDA response. Numeric outliers use the profiled IQR rule. Correlations describe association, not causation.",
  ].filter(Boolean).join("\n");

  downloadBrandedPdf({
    filename: `${base}_eda_report.pdf`,
    title: "Exploratory Data Analysis",
    subtitle: "Data quality, distributions, relationships, and next steps",
    datasetName: dataset.filename,
    markdown,
    metrics: [
      { label: "Rows", value: dataset.rows.toLocaleString() },
      { label: "Columns", value: dataset.columns.toLocaleString() },
      { label: "Health", value: `${profile.health_score ?? 0}/100` },
    ],
    charts,
  });
}

export function downloadModelingReportPdf(data: ModelingPdfData) {
  const successful = data.results.filter((row) => !row.error);
  const failed = data.results.filter((row) => row.error);
  const sorted = [...successful].sort(data.taskType === "regression"
    ? (a, b) => (a.rmse ?? Number.POSITIVE_INFINITY) - (b.rmse ?? Number.POSITIVE_INFINITY)
    : (a, b) => (b.f1 ?? b.primary_score ?? 0) - (a.f1 ?? a.primary_score ?? 0));
  const best = sorted[0];
  const bestMetric = best ? primaryMetric(best, data.taskType) : "Not available";
  const base = safeBaseName(data.dataset.filename);
  const leaderboard = data.taskType === "regression"
    ? ["| Rank | Model | RMSE | MAE | R2 | CV score |", "| ---: | --- | ---: | ---: | ---: | ---: |", ...sorted.map((row, index) => `| ${index + 1} | ${md(prettyModel(row.model_name))} | ${formatNumber(row.rmse)} | ${formatNumber(row.mae)} | ${formatNumber(row.r2)} | ${formatNumber(row.cv_score)} |`)]
    : ["| Rank | Model | F1 | Accuracy | Precision | Recall | ROC-AUC | CV score |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |", ...sorted.map((row, index) => `| ${index + 1} | ${md(prettyModel(row.model_name))} | ${formatNumber(row.f1)} | ${formatNumber(row.accuracy)} | ${formatNumber(row.precision)} | ${formatNumber(row.recall)} | ${formatNumber(row.roc_auc)} | ${formatNumber(row.cv_score)} |`)];
  const charts = buildModelCharts(sorted, data.taskType);

  const markdown = [
    "## Training Summary",
    `DataStory AI trained **${successful.length} successful model${successful.length === 1 ? "" : "s"}** for a **${capitalize(data.taskType || "undetected")}** task using **${data.target || "no selected target"}** as the target.`,
    best ? `The current leader is **${prettyModel(best.model_name)}** with **${bestMetric}**.` : "No successful training run is available yet. This sample report will populate after models are trained.",
    "",
    "## Experiment Configuration",
    "| Setting | Value |",
    "| --- | --- |",
    `| Dataset | ${md(data.dataset.filename)} |`,
    `| Target | ${md(data.target || "Not selected")} |`,
    `| Task type | ${capitalize(data.taskType || "Not detected")} |`,
    `| Train / test split | ${Math.round((1 - data.testSize) * 100)}% / ${Math.round(data.testSize * 100)}% |`,
    `| Cross-validation | ${data.cvFolds}-fold |`,
    `| Imbalance strategy | ${capitalize((data.imbalanceStrategy || "none").replace(/_/g, " "))} |`,
    `| Primary metric | ${data.taskType === "regression" ? "RMSE (lower is better)" : "Weighted F1 (higher is better)"} |`,
    "",
    "## Model Leaderboard",
    ...(leaderboard.length > 2 ? leaderboard : ["No model metrics are available yet."]),
    "",
    "## Best Model",
    best ? `**${prettyModel(best.model_name)}** leads the current comparison. ${best.reason || "Treat this result as a benchmark and confirm stability with repeated validation before deployment."}` : "Train at least one model to populate the winner analysis.",
    ...(best?.best_params && Object.keys(best.best_params).length ? ["", "### Best Parameters", ...Object.entries(best.best_params).map(([key, value]) => `- ${key}: ${String(value)}`)] : []),
    "",
    ...(failed.length ? ["## Failed Runs", ...failed.map((row) => `- ${prettyModel(row.model_name)}: ${row.error || "Training failed"}`), ""] : []),
    "## Interpretation Notes",
    `- Compare models on the same held-out split and validation settings.`,
    `- ${data.taskType === "regression" ? "Lower RMSE and MAE are better; inspect R2 for explained variance." : "Balance weighted F1 with class-level recall, precision, and ROC-AUC where available."}`,
    "- Confirm there is no target leakage before treating the leaderboard as final.",
    "- Re-run training after material cleaning or feature changes and retain the report with the dataset version.",
    "",
    "## Technical Notes",
    "Metrics shown here come from the active DataStory training session. A strong test result is not a deployment guarantee; validate on representative future data and monitor drift.",
  ].join("\n");

  downloadBrandedPdf({
    filename: `${base}_modeling_report.pdf`,
    title: "Modeling Report",
    subtitle: "Training configuration, model leaderboard, and interpretation",
    datasetName: data.dataset.filename,
    markdown,
    metrics: [
      { label: "Models", value: successful.length.toString() },
      { label: "Best model", value: best ? prettyModel(best.model_name) : "Pending" },
      { label: "Best score", value: bestMetric },
    ],
    charts,
  });
}

export function downloadMarkdownReportPdf(options: Omit<PdfExportOptions, "metrics"> & { metrics?: PdfExportOptions["metrics"] }) {
  downloadBrandedPdf(options);
}

function downloadBrandedPdf(options: PdfExportOptions) {
  const doc = new jsPDF({ unit: "mm", format: "a4", compress: true });
  drawCover(doc, options);
  if (options.charts?.length) renderChartAppendix(doc, options.charts, options.datasetName);
  doc.addPage();
  drawContentPageChrome(doc, options.datasetName);
  renderMarkdown(doc, options.markdown, options.datasetName);
  addPageNumbers(doc);
  doc.setProperties({
    title: options.title,
    subject: options.subtitle,
    author: "DataStory AI",
    creator: "DataStory AI",
    keywords: "data analysis, EDA, machine learning, report",
  });
  doc.save(options.filename);
}

function renderChartAppendix(doc: jsPDF, charts: PdfChart[], datasetName: string) {
  let y = 0;
  charts.forEach((chart, index) => {
    const chartHeight = chart.type === "heatmap" ? 205 : 112;
    if (index === 0 || y + chartHeight > 276) {
      doc.addPage();
      drawContentPageChrome(doc, datasetName);
      y = 31;
      drawSectionTitle(doc, index === 0 ? "Visual Analysis" : "Visual Analysis - Continued", y);
      y += 18;
    }
    drawChartCard(doc, chart, 18, y, 174, chartHeight - 8);
    y += chartHeight;
  });
}

function drawSectionTitle(doc: jsPDF, title: string, y: number) {
  doc.setFillColor(...COLORS.sagePale);
  doc.roundedRect(18, y - 7, 174, 13, 2, 2, "F");
  doc.setTextColor(...COLORS.sageDark);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(17);
  doc.text(title, 23, y + 1.5);
}

function drawChartCard(doc: jsPDF, chart: PdfChart, x: number, y: number, width: number, height: number) {
  doc.setFillColor(250, 251, 248);
  doc.setDrawColor(...COLORS.line);
  doc.roundedRect(x, y, width, height, 3, 3, "FD");
  doc.setTextColor(...COLORS.ink);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(12);
  doc.text(truncate(chart.title, 74), x + 7, y + 10);
  if (chart.subtitle) {
    doc.setFont("helvetica", "normal");
    doc.setTextColor(...COLORS.muted);
    doc.setFontSize(7.5);
    doc.text(truncate(chart.subtitle, 96), x + 7, y + 16);
  }
  const top = y + (chart.subtitle ? 23 : 18);
  const innerHeight = height - (chart.subtitle ? 29 : 24);
  if (chart.type === "donut") drawDonutChart(doc, chart, x + 7, top, width - 14, innerHeight);
  if (chart.type === "bar") drawBarChart(doc, chart, x + 7, top, width - 14, innerHeight);
  if (chart.type === "grouped") drawGroupedChart(doc, chart, x + 7, top, width - 14, innerHeight);
  if (chart.type === "heatmap") drawHeatmap(doc, chart, x + 7, top, width - 14, innerHeight);
}

function drawDonutChart(doc: jsPDF, chart: Extract<PdfChart, { type: "donut" }>, x: number, y: number, width: number, height: number) {
  const total = Math.max(1, sum(chart.values));
  const cx = x + Math.min(width * 0.34, 50);
  const cy = y + height / 2;
  const radius = Math.min(25, height * 0.38);
  let start = -90;
  chart.values.forEach((value, index) => {
    const angle = (Math.max(0, value) / total) * 360;
    drawSector(doc, cx, cy, radius, start, start + angle, chartColor(index));
    start += angle;
  });
  doc.setFillColor(250, 251, 248);
  doc.circle(cx, cy, radius * 0.55, "F");
  doc.setTextColor(...COLORS.ink);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(12);
  doc.text(formatCompact(total), cx, cy + 1, { align: "center" });
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7);
  doc.text("TOTAL", cx, cy + 6, { align: "center" });
  const legendX = x + Math.min(width * 0.64, 98);
  chart.labels.slice(0, 7).forEach((label, index) => {
    const rowY = y + 8 + index * 9;
    doc.setFillColor(...chartColor(index));
    doc.roundedRect(legendX, rowY - 3.2, 4, 4, 0.7, 0.7, "F");
    doc.setTextColor(...COLORS.ink);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(7.5);
    doc.text(`${truncate(label, 28)}  ${formatCompact(chart.values[index] ?? 0)} (${formatNumber(((chart.values[index] ?? 0) / total) * 100)}%)`, legendX + 7, rowY);
  });
}

function drawSector(doc: jsPDF, cx: number, cy: number, radius: number, start: number, end: number, color: [number, number, number]) {
  const steps = Math.max(3, Math.ceil(Math.abs(end - start) / 7));
  const points: Array<[number, number]> = [[cx, cy]];
  for (let index = 0; index <= steps; index += 1) {
    const angle = ((start + ((end - start) * index) / steps) * Math.PI) / 180;
    points.push([cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius]);
  }
  doc.setFillColor(...color);
  const [first, ...rest] = points;
  doc.lines(rest.map((point, index) => index === 0 ? [point[0] - first[0], point[1] - first[1]] : [point[0] - points[index][0], point[1] - points[index][1]]), first[0], first[1], [1, 1], "F", true);
}

function drawBarChart(doc: jsPDF, chart: Extract<PdfChart, { type: "bar" }>, x: number, y: number, width: number, height: number) {
  const entries = chart.labels.map((label, index) => ({ label, value: Number(chart.values[index] ?? 0) })).slice(0, chart.horizontal ? 9 : 12);
  const max = Math.max(...entries.map((entry) => Math.abs(entry.value)), 1);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(6.7);
  if (chart.horizontal !== false) {
    const labelWidth = Math.min(43, width * 0.32);
    const barWidth = width - labelWidth - 25;
    const rowHeight = height / Math.max(1, entries.length);
    entries.forEach((entry, index) => {
      const rowY = y + index * rowHeight + rowHeight * 0.62;
      doc.setTextColor(...COLORS.muted);
      doc.text(truncate(entry.label, 25), x + labelWidth - 2, rowY, { align: "right" });
      doc.setFillColor(231, 236, 228);
      doc.roundedRect(x + labelWidth, rowY - 3.3, barWidth, 4.5, 1.4, 1.4, "F");
      doc.setFillColor(...chartColor(index));
      doc.roundedRect(x + labelWidth, rowY - 3.3, Math.max(1.5, (Math.abs(entry.value) / max) * barWidth), 4.5, 1.4, 1.4, "F");
      doc.setTextColor(...COLORS.ink);
      doc.text(`${formatCompact(entry.value)}${chart.valueSuffix ?? ""}`, x + labelWidth + barWidth + 3, rowY);
    });
    return;
  }
  const baseline = y + height - 15;
  const columnWidth = width / Math.max(1, entries.length);
  entries.forEach((entry, index) => {
    const barHeight = (Math.abs(entry.value) / max) * (height - 27);
    const barX = x + index * columnWidth + columnWidth * 0.18;
    doc.setFillColor(...chartColor(index));
    doc.roundedRect(barX, baseline - barHeight, columnWidth * 0.64, barHeight, 1.2, 1.2, "F");
    doc.setTextColor(...COLORS.ink);
    doc.text(formatCompact(entry.value), barX + columnWidth * 0.32, baseline - barHeight - 2, { align: "center" });
    doc.setTextColor(...COLORS.muted);
    doc.text(truncate(entry.label, 12), barX + columnWidth * 0.32, baseline + 5, { align: "center", angle: entries.length > 7 ? 35 : 0 });
  });
}

function drawGroupedChart(doc: jsPDF, chart: Extract<PdfChart, { type: "grouped" }>, x: number, y: number, width: number, height: number) {
  const labels = chart.labels.slice(0, 7);
  const series = chart.series.slice(0, 5);
  const max = Math.max(...series.flatMap((item) => item.values.slice(0, labels.length).map((value) => Math.abs(Number(value)))), 1);
  const legendHeight = 9;
  series.forEach((item, index) => {
    const legendX = x + index * Math.min(34, width / Math.max(1, series.length));
    doc.setFillColor(...chartColor(index));
    doc.rect(legendX, y + 1, 4, 3, "F");
    doc.setTextColor(...COLORS.muted);
    doc.setFontSize(6.5);
    doc.text(truncate(item.name, 15), legendX + 6, y + 4);
  });
  const baseline = y + height - 14;
  const groupWidth = width / Math.max(1, labels.length);
  const barWidth = Math.max(1.3, (groupWidth * 0.72) / Math.max(1, series.length));
  labels.forEach((label, labelIndex) => {
    series.forEach((item, seriesIndex) => {
      const value = Math.abs(Number(item.values[labelIndex] ?? 0));
      const barHeight = (value / max) * (height - legendHeight - 25);
      const barX = x + labelIndex * groupWidth + groupWidth * 0.14 + seriesIndex * barWidth;
      doc.setFillColor(...chartColor(seriesIndex));
      doc.rect(barX, baseline - barHeight, Math.max(1, barWidth - 0.5), barHeight, "F");
    });
    doc.setTextColor(...COLORS.muted);
    doc.setFontSize(6.3);
    doc.text(truncate(label, 13), x + labelIndex * groupWidth + groupWidth / 2, baseline + 5, { align: "center", angle: labels.length > 5 ? 28 : 0 });
  });
}

function drawHeatmap(doc: jsPDF, chart: Extract<PdfChart, { type: "heatmap" }>, x: number, y: number, width: number, height: number) {
  const labels = chart.labels.slice(0, 12);
  const size = Math.min(width - 40, height - 22);
  const cell = size / Math.max(1, labels.length);
  const gridX = x + 28;
  const gridY = y + 2;
  labels.forEach((row, rowIndex) => {
    labels.forEach((_column, columnIndex) => {
      const value = Number(chart.values[rowIndex]?.[columnIndex] ?? 0);
      doc.setFillColor(...heatColor(value));
      doc.rect(gridX + columnIndex * cell, gridY + rowIndex * cell, cell + 0.15, cell + 0.15, "F");
      if (cell >= 8) {
        doc.setTextColor(Math.abs(value) > 0.55 ? 255 : 50, Math.abs(value) > 0.55 ? 255 : 60, Math.abs(value) > 0.55 ? 255 : 52);
        doc.setFontSize(Math.min(6, cell * 0.42));
        doc.text(value.toFixed(2), gridX + columnIndex * cell + cell / 2, gridY + rowIndex * cell + cell * 0.62, { align: "center" });
      }
    });
    doc.setTextColor(...COLORS.muted);
    doc.setFontSize(6.2);
    doc.text(truncate(row, 13), gridX - 2, gridY + rowIndex * cell + cell * 0.63, { align: "right" });
    doc.text(truncate(row, 13), gridX + rowIndex * cell + cell * 0.5, gridY + size + 5, { align: "right", angle: 45 });
  });
  doc.setTextColor(...COLORS.muted);
  doc.setFontSize(7);
  doc.text("-1 negative", gridX, gridY + size + 18);
  doc.text("0 neutral", gridX + size / 2, gridY + size + 18, { align: "center" });
  doc.text("+1 positive", gridX + size, gridY + size + 18, { align: "right" });
}

function drawCover(doc: jsPDF, options: PdfExportOptions) {
  doc.setFillColor(...COLORS.ink);
  doc.rect(0, 0, 210, 297, "F");
  doc.setFillColor(...COLORS.sage);
  doc.circle(178, 36, 42, "F");
  doc.setFillColor(117, 151, 101);
  doc.circle(188, 24, 25, "F");
  doc.setDrawColor(...COLORS.sagePale);
  doc.setLineWidth(0.7);
  doc.circle(28, 30, 10, "S");
  doc.setTextColor(...COLORS.sagePale);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.text("DS", 28, 32, { align: "center" });
  doc.setFontSize(10);
  doc.text("DATASTORY AI", 44, 33);

  doc.setTextColor(...COLORS.white);
  doc.setFontSize(30);
  doc.text(doc.splitTextToSize(options.title, 155), 24, 112);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(211, 224, 207);
  doc.setFontSize(13);
  doc.text(doc.splitTextToSize(options.subtitle, 145), 24, 139);
  doc.setDrawColor(...COLORS.sage);
  doc.setLineWidth(1.2);
  doc.line(24, 157, 78, 157);
  doc.setFontSize(10);
  doc.setTextColor(178, 195, 176);
  doc.text("DATASET", 24, 177);
  doc.setTextColor(...COLORS.white);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(15);
  doc.text(doc.splitTextToSize(options.datasetName, 150), 24, 188);

  if (options.metrics?.length) {
    const boxWidth = 49;
    options.metrics.slice(0, 3).forEach((metric, index) => {
      const x = 24 + index * 54;
      doc.setFillColor(38, 51, 41);
      doc.roundedRect(x, 218, boxWidth, 34, 3, 3, "F");
      doc.setTextColor(172, 192, 168);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(8);
      doc.text(metric.label.toUpperCase(), x + 5, 229);
      doc.setTextColor(...COLORS.white);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(metric.value.length > 18 ? 10 : 14);
      doc.text(doc.splitTextToSize(metric.value, boxWidth - 10).slice(0, 2), x + 5, 241);
    });
  }
  doc.setTextColor(151, 171, 150);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.text(`Generated ${new Date().toLocaleString()}  |  Confidential analysis workspace`, 24, 278);
}

function renderMarkdown(doc: jsPDF, markdown: string, datasetName: string) {
  const lines = markdown.replace(/\r/g, "").split("\n");
  let y = 31;
  let index = 0;
  while (index < lines.length) {
    const raw = lines[index].trim();
    if (!raw) {
      y += 2.5;
      index += 1;
      continue;
    }
    if (raw.startsWith("```")) {
      const language = raw.slice(3).trim().toLowerCase();
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) index += 1;
      if (index < lines.length) index += 1;
      if (language === "mermaid") {
        y = ensureSpace(doc, y, 12, datasetName);
        doc.setFillColor(...COLORS.sagePale);
        doc.roundedRect(18, y - 5, 174, 10, 2, 2, "F");
        doc.setTextColor(...COLORS.sageDark);
        doc.setFont("helvetica", "italic");
        doc.setFontSize(8.5);
        doc.text("Chart rendered in the Visual Analysis section of this report.", 23, y + 1.2);
        y += 11;
      }
      continue;
    }
    if (raw.startsWith("|")) {
      const tableLines: string[] = [];
      while (index < lines.length && lines[index].trim().startsWith("|")) {
        tableLines.push(lines[index].trim());
        index += 1;
      }
      const rows = tableLines.map(parseTableRow).filter((row) => row.length);
      const body = rows.slice(1).filter((row) => !row.every((cell) => /^:?-{3,}:?$/.test(cell)));
      ensureSpace(doc, y, 24, datasetName);
      autoTable(doc, {
        startY: y,
        head: rows.length ? [rows[0]] : [],
        body,
        margin: { left: 18, right: 18, top: 28, bottom: 20 },
        theme: "grid",
        styles: { font: "helvetica", fontSize: rows[0]?.length > 6 ? 6.7 : 8, cellPadding: 2.3, textColor: COLORS.ink, lineColor: COLORS.line, lineWidth: 0.2, overflow: "linebreak" },
        headStyles: { fillColor: COLORS.sageDark, textColor: COLORS.white, fontStyle: "bold" },
        alternateRowStyles: { fillColor: [247, 249, 245] },
        didDrawPage: () => drawContentPageChrome(doc, datasetName),
      });
      y = ((doc as jsPDF & { lastAutoTable?: { finalY: number } }).lastAutoTable?.finalY ?? y) + 7;
      continue;
    }
    if (raw.startsWith("#")) {
      const level = raw.match(/^#+/)?.[0].length ?? 2;
      const title = cleanMarkdown(raw.replace(/^#+\s*/, ""));
      const fontSize = level <= 2 ? 17 : 12;
      const required = level <= 2 ? 19 : 13;
      y = ensureSpace(doc, y, required, datasetName);
      if (level <= 2) {
        doc.setFillColor(...COLORS.sagePale);
        doc.roundedRect(18, y - 7, 174, 13, 2, 2, "F");
        doc.setTextColor(...COLORS.sageDark);
        doc.setFont("helvetica", "bold");
        doc.setFontSize(fontSize);
        doc.text(title, 23, y + 1.5);
        y += 14;
      } else {
        doc.setTextColor(...COLORS.sageDark);
        doc.setFont("helvetica", "bold");
        doc.setFontSize(fontSize);
        doc.text(title, 18, y);
        y += 8;
      }
      index += 1;
      continue;
    }
    if (/^[-*]\s+/.test(raw)) {
      const item = cleanMarkdown(raw.replace(/^[-*]\s+/, ""));
      const wrapped = doc.splitTextToSize(item, 162) as string[];
      y = ensureSpace(doc, y, wrapped.length * 4.5 + 4, datasetName);
      doc.setFillColor(...COLORS.sage);
      doc.circle(22, y - 1.1, 1, "F");
      doc.setTextColor(...COLORS.ink);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(9.5);
      doc.text(wrapped, 27, y);
      y += wrapped.length * 4.5 + 3;
      index += 1;
      continue;
    }
    const paragraph: string[] = [raw];
    index += 1;
    while (index < lines.length) {
      const next = lines[index].trim();
      if (!next || next.startsWith("#") || next.startsWith("|") || next.startsWith("```") || /^[-*]\s+/.test(next)) break;
      paragraph.push(next);
      index += 1;
    }
    const wrapped = doc.splitTextToSize(cleanMarkdown(paragraph.join(" ")), 174) as string[];
    y = ensureSpace(doc, y, wrapped.length * 4.7 + 4, datasetName);
    doc.setTextColor(...COLORS.ink);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9.5);
    doc.text(wrapped, 18, y);
    y += wrapped.length * 4.7 + 4;
  }
}

function ensureSpace(doc: jsPDF, y: number, required: number, datasetName: string) {
  if (y + required <= 276) return y;
  doc.addPage();
  drawContentPageChrome(doc, datasetName);
  return 31;
}

function drawContentPageChrome(doc: jsPDF, datasetName: string) {
  doc.saveGraphicsState();
  doc.setFillColor(...COLORS.ink);
  doc.rect(0, 0, 210, 18, "F");
  doc.setTextColor(...COLORS.sagePale);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(8);
  doc.text("DATASTORY AI", 18, 11.5);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(194, 207, 190);
  doc.text(truncate(datasetName, 58), 105, 11.5, { align: "center" });
  doc.setDrawColor(...COLORS.line);
  doc.line(18, 283, 192, 283);
  doc.setTextColor(...COLORS.muted);
  doc.setFontSize(7.5);
  doc.text("Generated by DataStory AI", 18, 289);
  doc.setTextColor(230, 237, 227);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(13);
  doc.text("DATASTORY AI", 203, 168, { align: "center", angle: 90 });
  doc.restoreGraphicsState();
}

function addPageNumbers(doc: jsPDF) {
  const pages = doc.getNumberOfPages();
  for (let page = 2; page <= pages; page += 1) {
    doc.setPage(page);
    doc.saveGraphicsState();
    doc.setTextColor(...COLORS.muted);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(7.5);
    doc.text(`Page ${page - 1} of ${pages - 1}`, 105, 289, { align: "center" });
    doc.restoreGraphicsState();
  }
}

function parseTableRow(line: string) {
  return line.slice(1, line.endsWith("|") ? -1 : undefined).split("|").map((cell) => cleanMarkdown(cell.trim()));
}

function cleanMarkdown(value: string) {
  return value.replace(/\\\|/g, "|").replace(/\*\*/g, "").replace(/`/g, "").replace(/\s+/g, " ").trim();
}

function strongestCorrelations(correlation?: { columns: string[]; values: number[][] }) {
  if (!correlation?.columns?.length) return [];
  const rows: Array<{ pair: string; value: number }> = [];
  correlation.columns.forEach((left, rowIndex) => correlation.columns.forEach((right, columnIndex) => {
    if (columnIndex <= rowIndex) return;
    const value = Number(correlation.values[rowIndex]?.[columnIndex]);
    if (Number.isFinite(value)) rows.push({ pair: `${left} vs ${right}`, value });
  }));
  return rows.sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 8);
}

function buildEdaCharts(dataset: DatasetResponse, eda: EdaResponse | undefined, missing: Array<{ name: string; pct: number }>): PdfChart[] {
  const profile = dataset.profile;
  const charts: PdfChart[] = [
    {
      type: "donut",
      title: "Column Data Types",
      subtitle: "The same feature-type composition shown in the EDA workspace",
      labels: ["Numeric", "Categorical", "Date", "Boolean"],
      values: [profile.numeric_cols ?? 0, profile.categorical_cols ?? 0, profile.date_cols ?? 0, profile.bool_cols ?? 0],
    },
  ];
  if (missing.length) {
    charts.push({ type: "bar", title: "Missing Values by Column", subtitle: "Top columns by missing percentage", labels: missing.map((row) => row.name), values: missing.map((row) => row.pct), valueSuffix: "%" });
  } else {
    charts.push({ type: "donut", title: "Data Completeness", subtitle: "Profiled cells split into complete and missing", labels: ["Complete", "Missing"], values: [Math.max(0, dataset.rows * dataset.columns - (profile.total_missing ?? 0)), profile.total_missing ?? 0] });
  }
  (eda?.numeric_profiles ?? []).slice(0, 2).forEach((item) => {
    const labels = item.histogram.bins.slice(0, -1).map((value, index) => formatRange(value, item.histogram.bins[index + 1]));
    charts.push({ type: "bar", title: `${item.column} Distribution`, subtitle: "Histogram from the generated EDA profile", labels, values: item.histogram.counts, horizontal: false });
  });
  const categorical = eda?.categorical_profiles?.[0];
  if (categorical) charts.push({ type: "bar", title: `${categorical.column} Categories`, subtitle: "Top generated category counts", labels: categorical.labels.slice(0, 10), values: categorical.counts.slice(0, 10) });
  if (eda?.correlation?.columns?.length && eda.correlation.columns.length > 1) {
    charts.push({ type: "heatmap", title: "Correlation Matrix", subtitle: "Pearson correlation across numeric features", labels: eda.correlation.columns, values: eda.correlation.values });
  }
  return charts;
}

function buildModelCharts(results: TrainResult[], taskType: string): PdfChart[] {
  if (!results.length) return [];
  const models = results.slice(0, 8);
  const labels = models.map((row) => prettyModel(row.model_name));
  const charts: PdfChart[] = [
    {
      type: "bar",
      title: "Model Leaderboard",
      subtitle: taskType === "regression" ? "RMSE by model - lower is better" : "Weighted F1 by model - higher is better",
      labels,
      values: models.map((row) => Number(taskType === "regression" ? row.rmse : row.f1 ?? row.primary_score ?? 0)),
    },
  ];
  charts.push(taskType === "regression"
    ? { type: "grouped", title: "Regression Metric Comparison", subtitle: "Error and explained variance for each trained model", labels, series: [
        { name: "RMSE", values: models.map((row) => Number(row.rmse ?? 0)) },
        { name: "MAE", values: models.map((row) => Number(row.mae ?? 0)) },
        { name: "R2", values: models.map((row) => Number(row.r2 ?? 0)) },
      ] }
    : { type: "grouped", title: "Classification Metric Comparison", subtitle: "Accuracy, precision, recall, F1, and ROC-AUC", labels, series: [
        { name: "Accuracy", values: models.map((row) => Number(row.accuracy ?? 0)) },
        { name: "Precision", values: models.map((row) => Number(row.precision ?? 0)) },
        { name: "Recall", values: models.map((row) => Number(row.recall ?? 0)) },
        { name: "F1", values: models.map((row) => Number(row.f1 ?? row.primary_score ?? 0)) },
        { name: "ROC-AUC", values: models.map((row) => Number(row.roc_auc ?? 0)) },
      ] });
  return charts;
}

function chartColor(index: number): [number, number, number] {
  const palette: Array<[number, number, number]> = [COLORS.sage, [79, 131, 232], [240, 182, 110], [154, 112, 181], [217, 70, 70], [88, 160, 156], [184, 150, 70]];
  return palette[index % palette.length];
}

function heatColor(value: number): [number, number, number] {
  const amount = Math.min(1, Math.abs(value));
  const neutral: [number, number, number] = [248, 247, 242];
  const target: [number, number, number] = value < 0 ? [79, 131, 232] : [217, 70, 70];
  return neutral.map((channel, index) => Math.round(channel + (target[index] - channel) * amount)) as [number, number, number];
}

function sum(values: number[]) {
  return values.reduce((total, value) => total + Math.max(0, Number(value) || 0), 0);
}

function formatCompact(value: number) {
  const absolute = Math.abs(value);
  if (absolute >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}m`;
  if (absolute >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
}

function formatRange(left: number, right: number) {
  return `${formatCompact(left)}-${formatCompact(right)}`;
}

function primaryMetric(row: TrainResult, taskType: string) {
  return taskType === "regression" ? `RMSE ${formatNumber(row.rmse)}` : `Weighted F1 ${formatNumber(row.f1 ?? row.primary_score)}`;
}

function prettyModel(value: string) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function safeBaseName(filename: string) {
  return filename.replace(/\.csv$/i, "").replace(/[^a-z0-9_-]+/gi, "_").replace(/^_+|_+$/g, "") || "dataset";
}

function md(value: unknown) {
  return String(value ?? "-").replace(/\|/g, "\\|").replace(/\r?\n/g, " ");
}

function formatWhole(value?: number) {
  return typeof value === "number" ? value.toLocaleString() : "0";
}

function formatNumber(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString(undefined, { maximumFractionDigits: 3 }) : "-";
}

function capitalize(value: string) {
  return value ? value.charAt(0).toUpperCase() + value.slice(1) : value;
}

function truncate(value: string, length: number) {
  return value.length <= length ? value : `${value.slice(0, length - 3)}...`;
}
