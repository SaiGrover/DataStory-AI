import Plot from "react-plotly.js";
import { useState } from "react";
import { useQuery } from "react-query";
import { useNavigate } from "react-router-dom";
import type { LucideIcon } from "lucide-react";
import { AlertCircle, ArrowRight, CalendarDays, Copy, Database, Eye, Lightbulb, MessageSquare, ShieldCheck, Wand2 } from "lucide-react";
import { getEda } from "../lib/api";
import { useAppState } from "../lib/store";

function EmptyState() {
  return <div className="ds-card p-8">No dataset loaded. Go to Start and upload a CSV or load a sample dataset.</div>;
}

export function HealthPage() {
  const { dataset } = useAppState();
  const navigate = useNavigate();
  const [detailView, setDetailView] = useState("summary");
  const [chartColumn, setChartColumn] = useState("");
  const eda = useQuery(["eda", dataset?.dataset_id], () => getEda(dataset!.dataset_id), { enabled: Boolean(dataset) });
  if (!dataset) return <EmptyState />;

  const profile = dataset.profile;
  const details = profile.column_details ?? [];
  const missingCols = details
    .map((d) => ({ col: String(d.Column ?? ""), pct: Number(d["Null %"] ?? 0) }))
    .sort((a, b) => b.pct - a.pct)
    .slice(0, 7);
  const score = profile.health_score ?? 0;
  const memory = Math.max(1, Math.round(JSON.stringify(profile).length / 10));
  const missingPct = Math.round(((profile.total_missing ?? 0) / Math.max(1, dataset.rows * dataset.columns)) * 1000) / 10;

  const issueCards: Array<[string, number, string, LucideIcon, string]> = [
    ["Missing Values", profile.total_missing ?? 0, "cells affected", AlertCircle, "bg-[#EEF5E9]"],
    ["Duplicate Rows", profile.duplicates ?? 0, "duplicates found", Copy, "bg-[#FBF3DF]"],
    ["High Cardinality", profile.high_cardinality_cols?.length ?? 0, "columns affected", AlertCircle, "bg-[#FBEDE5]"],
    ["Constant Columns", profile.constant_cols?.length ?? 0, "constant columns", ShieldCheck, "bg-[#F4EDF8]"],
    ["Possible Date Columns", profile.date_cols ?? 0, "columns detected", CalendarDays, "bg-[#ECF4F8]"],
  ];

  const summaryRows = [
    ["Total Rows", dataset.rows.toLocaleString()],
    ["Total Columns", dataset.columns.toLocaleString()],
    ["Missing Values", `${(profile.total_missing ?? 0).toLocaleString()} (${missingPct}%)`],
    ["Duplicate Rows", (profile.duplicates ?? 0).toLocaleString()],
    ["Numeric Columns", profile.numeric_cols ?? 0],
    ["Categorical Columns", profile.categorical_cols ?? 0],
    ["Memory Usage", `${memory} KB`],
  ];

  const recommendations = [
    "Handle missing values before cleaning.",
    "Review duplicate rows before training.",
    "Select a target column to check class balance.",
    "Check high-cardinality fields for encoding.",
  ];
  const numericProfiles = eda.data?.numeric_profiles ?? [];
  const categoricalProfiles = eda.data?.categorical_profiles ?? [];
  const selectedNumericProfile = numericProfiles.find((item) => item.column === chartColumn);
  const selectedCategoricalProfile = categoricalProfiles.find((item) => item.column === chartColumn);
  const activeNumericProfile = selectedNumericProfile ?? (!chartColumn ? numericProfiles[0] : undefined);
  const activeCategoricalProfile = selectedCategoricalProfile ?? (!chartColumn && !numericProfiles.length ? categoricalProfiles[0] : undefined);
  const histogramX = activeNumericProfile
    ? activeNumericProfile.histogram.bins.slice(0, -1).map((bin, index) => Math.round(((bin + activeNumericProfile.histogram.bins[index + 1]) / 2) * 100) / 100)
    : [];

  const detailTitle: Record<string, string> = {
    summary: "Health Details",
    preview: "Preview Data",
    missing: "Missing Values Details",
    duplicates: "Duplicate Row Details",
    cardinality: "High Cardinality Details",
    constant: "Constant Column Details",
    dates: "Possible Date Column Details",
    missingReport: "Detailed Missing Report",
  };

  const detailRows = (() => {
    if (detailView === "preview") {
      return details.slice(0, 8).map((row) => ({
        label: String(row.Column),
        value: `${String(row.Type)} • ${String(row.Unique)} unique • top: ${String(row["Top Value"] ?? "-").slice(0, 28)}`,
      }));
    }
    if (detailView === "missing" || detailView === "missingReport") {
      return missingCols.map((item) => ({ label: item.col, value: `${item.pct}% missing` }));
    }
    if (detailView === "duplicates") {
      return [
        { label: "Duplicate rows", value: `${profile.duplicates ?? 0} exact duplicate rows detected` },
        { label: "Suggested action", value: "Use Clean Data > Duplicate Handling > Remove exact duplicate rows" },
      ];
    }
    if (detailView === "cardinality") {
      return (profile.high_cardinality_cols ?? []).length
        ? (profile.high_cardinality_cols ?? []).map((col) => ({ label: col, value: "High unique-value count" }))
        : [{ label: "High cardinality", value: "No high-cardinality columns detected" }];
    }
    if (detailView === "constant") {
      return (profile.constant_cols ?? []).length
        ? (profile.constant_cols ?? []).map((col) => ({ label: col, value: "Only one unique value" }))
        : [{ label: "Constant columns", value: "No constant columns detected" }];
    }
    if (detailView === "dates") {
      return [{ label: "Possible date columns", value: `${profile.date_cols ?? 0} columns detected as date-like` }];
    }
    return [
      { label: "Health score", value: `${score}/100` },
      { label: "Rows and columns", value: `${dataset.rows.toLocaleString()} rows • ${dataset.columns.toLocaleString()} columns` },
      { label: "Missing values", value: `${(profile.total_missing ?? 0).toLocaleString()} cells (${missingPct}%)` },
      { label: "Duplicate rows", value: `${profile.duplicates ?? 0}` },
    ];
  })();

  return (
    <div className="grid grid-cols-[250px_minmax(680px,1fr)_270px] gap-4 text-[13px]">
      <aside className="ds-card p-4">
        <h3 className="mb-3 text-sm font-extrabold">Dataset Overview</h3>
        <div className="mb-3 grid grid-cols-[38px_1fr_12px] items-center gap-3 rounded-lg bg-stone-50 p-3 dark:bg-zinc-800">
          <Database className="h-9 w-9 rounded-lg bg-[#EEF1EA] p-2 text-sage" />
          <div>
            <div className="text-sm font-extrabold">{dataset.filename}</div>
            <div className="text-xs text-zinc-500">{memory} KB</div>
          </div>
          <ArrowRight className="h-3 w-3 text-zinc-400" />
        </div>
        {[
          ["Rows", dataset.rows],
          ["Columns", dataset.columns],
          ["Numeric Columns", profile.numeric_cols ?? 0],
          ["Categorical Columns", profile.categorical_cols ?? 0],
        ].map(([label, value]) => (
          <div key={label} className="flex justify-between border-b border-line py-2 text-xs">
            <span className="text-zinc-500">{label}</span>
            <strong>{Number(value).toLocaleString()}</strong>
          </div>
        ))}
        <button
          className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg border border-line py-2 text-xs font-bold hover:bg-stone-50 dark:hover:bg-zinc-800"
          onClick={() => setDetailView("preview")}
        >
          <Eye className="h-3.5 w-3.5" />
          Preview Data
        </button>
      </aside>

      <section className="ds-card p-6">
        <div className="flex justify-between gap-6">
          <div>
            <h1 className="text-2xl font-black">Data Health Check</h1>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-300">We analyzed your dataset and generated the following insights.</p>
          </div>
          <div className="flex items-center gap-4">
            <ShieldCheck className="h-14 w-14 rounded-full bg-[#EEF1EA] p-3 text-sage" />
            <div>
              <div className="text-3xl font-black">
                {score} <span className="text-base text-zinc-500">/ 100</span>
              </div>
              <div className="text-xs text-zinc-500">Dataset Health Score</div>
            </div>
            <div className="grid h-16 w-16 place-items-center rounded-full" style={{ background: `conic-gradient(#839B63 ${score * 3.6}deg, #E8E2D7 0deg)` }}>
              <div className="h-11 w-11 rounded-full bg-white dark:bg-zinc-900" />
            </div>
          </div>
        </div>
        <div className="mt-6 grid grid-cols-5 gap-3">
          {issueCards.map(([label, value, caption, Icon, tint]) => {
            const detailKey = label === "Missing Values" ? "missing" : label === "Duplicate Rows" ? "duplicates" : label === "High Cardinality" ? "cardinality" : label === "Constant Columns" ? "constant" : "dates";
            return (
            <div key={label} className={`rounded-xl border border-line ${tint} p-4 dark:bg-zinc-800`}>
              <Icon className="mb-3 h-7 w-7 rounded-full bg-white p-1.5 text-sage dark:bg-zinc-900" />
              <div className="min-h-9 text-xs font-extrabold">{label}</div>
              <div className="mt-2 text-3xl font-black">{value.toLocaleString()}</div>
              <div className="text-xs text-zinc-500">{caption}</div>
              <button className="mt-4 text-xs font-bold text-sage hover:underline" onClick={() => setDetailView(detailKey)}>
                View details -&gt;
              </button>
            </div>
            );
          })}
        </div>
        <div className="mt-5 rounded-xl border border-line bg-white p-4 dark:bg-zinc-900">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-black">{detailTitle[detailView] ?? "Health Details"}</h3>
            <button className="text-xs font-bold text-sage hover:underline" onClick={() => navigate("/clean")}>
              Fix in Clean Data -&gt;
            </button>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2">
            {detailRows.map((row) => (
              <div key={`${row.label}-${row.value}`} className="rounded-lg bg-stone-50 px-3 py-2 text-xs dark:bg-zinc-800">
                <div className="font-bold">{row.label}</div>
                <div className="mt-1 text-zinc-500 dark:text-zinc-300">{row.value}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <aside className="ds-card p-4">
        <h3 className="mb-3 text-sm font-extrabold">Quick Summary</h3>
        {summaryRows.map(([label, value]) => (
          <div key={label} className="flex justify-between gap-4 border-b border-line py-2 text-xs">
            <span>{label}</span>
            <strong>{String(value)}</strong>
          </div>
        ))}
      </aside>

      <section className="ds-card col-span-2 p-5">
        <h3 className="mb-4 font-extrabold">Column Profile Summary</h3>
        <table className="w-full text-left text-xs">
          <thead className="bg-stone-50 text-zinc-500 dark:bg-zinc-800">
            <tr>
              <th className="p-2">Column</th>
              <th className="p-2">Data Type</th>
              <th className="p-2">Missing %</th>
              <th className="p-2">Unique</th>
              <th className="p-2">Sample Values</th>
              <th className="p-2">Issues</th>
            </tr>
          </thead>
          <tbody>
            {details.slice(0, 8).map((row, idx) => (
              <tr key={idx} className="border-b border-line">
                <td className="p-2 font-semibold">{String(row.Column)}</td>
                <td className="p-2">{String(row.Type)}</td>
                <td className="p-2">{String(row["Null %"])}%</td>
                <td className="p-2">{String(row.Unique)}</td>
                <td className="p-2">{String(row["Top Value"] ?? "-").slice(0, 26)}</td>
                <td className="p-2">
                  {Number(row["Null %"] ?? 0) > 0 ? <span className="rounded-full bg-[#F7E7EF] px-2 py-1 text-[10px] font-bold text-[#8A5270]">Missing Values</span> : "-"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="ds-card p-5">
        <h3 className="mb-4 font-extrabold">Missing Values Overview</h3>
        <Plot
          data={[{ type: "bar", x: missingCols.map((m) => m.col), y: missingCols.map((m) => m.pct), marker: { color: "#839B63" } }]}
          layout={{ height: 250, margin: { t: 10, r: 10, b: 70, l: 40 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent" }}
          config={{ displayModeBar: false, responsive: true }}
          className="w-full"
        />
        <button className="mt-1 text-xs font-bold text-sage hover:underline" onClick={() => setDetailView("missingReport")}>
          View detailed missing report -&gt;
        </button>
      </section>

      <section className="ds-card col-span-2 p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="font-extrabold">Column Diagnostics Before Cleaning</h3>
            <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-300">Check distribution, spread, and outlier signals before applying cleaning actions.</p>
          </div>
          <select
            className="rounded-lg border border-line bg-white px-3 py-2 text-xs font-bold dark:bg-zinc-900"
            value={chartColumn || activeNumericProfile?.column || activeCategoricalProfile?.column || ""}
            onChange={(event) => setChartColumn(event.target.value)}
          >
            {numericProfiles.map((item) => (
              <option key={item.column} value={item.column}>
                {item.column} - numeric
              </option>
            ))}
            {categoricalProfiles.map((item) => (
              <option key={item.column} value={item.column}>
                {item.column} - categorical
              </option>
            ))}
          </select>
        </div>
        {activeNumericProfile ? (
          <div className="mt-4 grid grid-cols-[1.25fr_.75fr] gap-4">
            <div className="rounded-xl border border-line p-3">
              <div className="text-xs font-black">Distribution: {activeNumericProfile.column}</div>
              <Plot
                data={[{ type: "bar", x: histogramX, y: activeNumericProfile.histogram.counts, marker: { color: "#839B63" } }]}
                layout={{ height: 230, margin: { t: 20, r: 12, b: 45, l: 38 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent" }}
                config={{ displayModeBar: false, responsive: true }}
                className="w-full"
              />
            </div>
            <div className="rounded-xl border border-line p-4">
              <div className="text-xs font-black">Outlier & Spread Check</div>
              <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                {[
                  ["Min", activeNumericProfile.box.min],
                  ["Q1", activeNumericProfile.box.q1],
                  ["Median", activeNumericProfile.box.median],
                  ["Q3", activeNumericProfile.box.q3],
                  ["Max", activeNumericProfile.box.max],
                  ["IQR Outliers", activeNumericProfile.box.outliers],
                ].map(([label, value]) => (
                  <div key={label} className="rounded-lg bg-stone-50 p-3 dark:bg-zinc-800">
                    <div className="font-bold text-zinc-500">{label}</div>
                    <div className="mt-1 text-lg font-black">{String(value)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : activeCategoricalProfile ? (
          <div className="mt-4 rounded-xl border border-line p-3">
            <div className="text-xs font-black">Category Balance: {activeCategoricalProfile.column}</div>
            <Plot
              data={[{ type: "bar", x: activeCategoricalProfile.labels, y: activeCategoricalProfile.counts, marker: { color: "#839B63" } }]}
              layout={{ height: 260, margin: { t: 20, r: 12, b: 70, l: 38 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent" }}
              config={{ displayModeBar: false, responsive: true }}
              className="w-full"
            />
          </div>
        ) : (
          <div className="mt-4 rounded-xl bg-stone-50 p-4 text-sm text-zinc-500 dark:bg-zinc-800">Diagnostics will appear after a dataset is loaded.</div>
        )}
        {(numericProfiles.length > 0 || categoricalProfiles.length > 0) && (
          <div className="mt-4 rounded-xl border border-line bg-stone-50 p-3 dark:bg-zinc-800">
            <div className="mb-2 text-xs font-black">All Column Checks</div>
            <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
              {numericProfiles.map((item) => (
                <button
                  key={item.column}
                  className={`rounded-lg border px-3 py-2 text-left text-xs transition ${
                    (chartColumn || numericProfiles[0]?.column) === item.column ? "border-sage bg-[#EEF5E9] text-sage" : "border-line bg-white hover:border-sage dark:bg-zinc-900"
                  }`}
                  onClick={() => setChartColumn(item.column)}
                >
                  <div className="truncate font-bold">{item.column}</div>
                  <div className="mt-1 text-zinc-500">numeric • {item.box.outliers} outliers</div>
                </button>
              ))}
              {categoricalProfiles.map((item) => (
                <button
                  key={item.column}
                  className={`rounded-lg border px-3 py-2 text-left text-xs transition ${
                    chartColumn === item.column ? "border-sage bg-[#EEF5E9] text-sage" : "border-line bg-white hover:border-sage dark:bg-zinc-900"
                  }`}
                  onClick={() => setChartColumn(item.column)}
                >
                  <div className="truncate font-bold">{item.column}</div>
                  <div className="mt-1 text-zinc-500">categorical • {item.labels.length} top values</div>
                </button>
              ))}
            </div>
          </div>
        )}
      </section>

      <section className="col-span-3 grid grid-cols-[250px_minmax(620px,1fr)_270px] gap-4">
        <aside className="ds-card p-4">
          <h3 className="mb-3 text-sm font-extrabold">Recommendations</h3>
          <ul className="space-y-3 text-zinc-600 dark:text-zinc-300">
            {recommendations.map((item) => (
              <li key={item} className="grid grid-cols-[20px_1fr] gap-2 text-xs">
                <ShieldCheck className="h-4 w-4 text-sage" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </aside>

        <section className="ds-card grid min-h-[190px] grid-cols-[150px_minmax(190px,.75fr)_1fr] items-center gap-5 overflow-hidden bg-[#FBF7ED] p-5 dark:bg-zinc-900">
          <img src="/pot_plant_small.png" alt="Potted plant" className="h-40 w-40 -translate-x-3 object-contain" />
          <div>
            <div className="mb-2 inline-grid h-9 w-9 place-items-center rounded-full bg-[#F2E4BD]">
              <Lightbulb className="h-4 w-4 text-sage" />
            </div>
            <h3 className="text-lg font-black">What's Next?</h3>
            <p className="mt-1 text-xs leading-5 text-zinc-600 dark:text-zinc-300">Your data health is ready. Clean and prepare your dataset for modeling.</p>
            <button className="ds-button-primary mt-3 px-3 py-2 text-xs" onClick={() => navigate("/clean")}>
              <Wand2 className="h-4 w-4" />
              Proceed to Cleaning
            </button>
          </div>
          <div className="grid grid-cols-[1fr_34px_1fr] items-center gap-3 text-xs">
            <div>
              <strong>Before cleaning</strong>
              <div>Rows {dataset.rows.toLocaleString()}</div>
              <div>Missing {(profile.total_missing ?? 0).toLocaleString()}</div>
              <div>Duplicates {profile.duplicates ?? 0}</div>
            </div>
            <ArrowRight className="h-7 w-7 text-zinc-300" />
            <div>
              <strong>After cleaning estimate</strong>
              <div>Rows {(dataset.rows - (profile.duplicates ?? 0)).toLocaleString()}</div>
              <div>Missing 0</div>
              <div>Duplicates 0</div>
            </div>
          </div>
        </section>

        <aside className="ds-card p-4">
          <h3 className="mb-3 text-sm font-extrabold">Need Help?</h3>
          <p className="text-xs text-zinc-600 dark:text-zinc-300">Ask DataStory Chat for suggestions or explanations about your data.</p>
          <button
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg border border-line py-2 text-xs font-bold hover:bg-stone-50 dark:hover:bg-zinc-800"
            onClick={() => navigate("/report")}
          >
            Open AI Report
            <MessageSquare className="h-3.5 w-3.5" />
          </button>
        </aside>
      </section>
    </div>
  );
}
