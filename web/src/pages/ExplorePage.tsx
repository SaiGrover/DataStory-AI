import Plot from "react-plotly.js";
import { useState } from "react";
import { useQuery } from "react-query";
import {
  AlertTriangle,
  BarChart3,
  ChevronRight,
  Download,
  FileText,
  Grid3X3,
  LineChart,
  PieChart,
  RefreshCw,
  Share2,
  Sparkles,
  Table2,
  Target,
} from "lucide-react";
import { getEda } from "../lib/api";
import { downloadTextFile, shareText } from "../lib/browserActions";
import { useAppState } from "../lib/store";

export function ExplorePage() {
  const { dataset } = useAppState();
  const [activeSection, setActiveSection] = useState("Overview");
  const [notice, setNotice] = useState("");
  const eda = useQuery(["eda", dataset?.dataset_id], () => getEda(dataset!.dataset_id), { enabled: Boolean(dataset) });
  if (!dataset) return <div className="ds-card p-8">No dataset loaded.</div>;

  const profile = dataset.profile;
  const details = profile.column_details ?? [];
  const numericCount = Number(profile.numeric_cols ?? 0);
  const categoricalCount = Number(profile.categorical_cols ?? 0);
  const missingRows = details
    .map((row) => ({ column: String(row.Column), count: Math.round((Number(row["Null %"] ?? 0) / 100) * dataset.rows) }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 12);
  const numericProfiles = eda.data?.numeric_profiles ?? [];
  const categoricalProfiles = eda.data?.categorical_profiles ?? [];
  const idColumns = eda.data?.id_columns ?? profile.id_cols ?? [];
  const idColumnSet = new Set(idColumns);
  const analysisNumericProfiles = numericProfiles.filter((item) => !idColumnSet.has(item.column));
  const analysisCategoricalProfiles = categoricalProfiles.filter((item) => !idColumnSet.has(item.column));
  const firstNumeric = analysisNumericProfiles.find((item) => item.column.toLowerCase() === "age") ?? analysisNumericProfiles[0];
  const secondNumeric = analysisNumericProfiles.find((item) => item.column.toLowerCase() === "fare") ?? analysisNumericProfiles.find((item) => item.column !== firstNumeric?.column) ?? analysisNumericProfiles[0];
  const uniqueByColumn = new Map(details.map((row) => [String(row.Column), Number(row.Unique ?? 0)]));
  const chartFriendlyCategoricals = analysisCategoricalProfiles.filter((item) => (uniqueByColumn.get(item.column) ?? item.labels.length) <= 20);
  const targetCandidates = (profile.possible_targets ?? []).filter((target) => !idColumnSet.has(target));
  const recommendedTarget = targetCandidates[0] ?? details.find((row) => !idColumnSet.has(String(row.Column)))?.Column?.toString() ?? "";
  const targetUnique = recommendedTarget ? uniqueByColumn.get(recommendedTarget) : undefined;
  const targetLabel = recommendedTarget
    ? `${recommendedTarget}${targetUnique === 2 ? " - Classification - Binary" : ""}`
    : "Not selected";
  const targetProfile = chartFriendlyCategoricals.find((item) => item.column === recommendedTarget);
  const firstCategorical = targetProfile ?? chartFriendlyCategoricals[0] ?? analysisCategoricalProfiles[0];
  const secondCategorical = chartFriendlyCategoricals.find((item) => item.column !== firstCategorical?.column) ?? chartFriendlyCategoricals[0] ?? analysisCategoricalProfiles[0];
  const missingTotal = Number(profile.total_missing ?? 0);
  const missingColumns = missingRows.filter((row) => row.count > 0);
  const correlationColumns = eda.data?.correlation?.columns ?? analysisNumericProfiles.map((item) => item.column);
  const correlationValues = eda.data?.correlation?.values ?? [];
  const topOutlierFeature = analysisNumericProfiles
    .filter((item) => item.box.outliers > 0)
    .sort((a, b) => b.box.outliers - a.box.outliers)[0];
  const keyFindings = [
    recommendedTarget ? `${recommendedTarget} is the recommended target column${targetUnique === 2 ? " for binary classification" : ""}.` : "No target column has been selected yet.",
    missingTotal === 0 ? "The current dataset has no missing cells." : `${missingTotal.toLocaleString()} missing cells still need attention.`,
    topOutlierFeature ? `${topOutlierFeature.column} contains ${topOutlierFeature.box.outliers.toLocaleString()} IQR outliers.` : "No IQR outliers were detected in the profiled numeric columns.",
    idColumns.length ? `${idColumns.join(", ")} identified as ID column${idColumns.length > 1 ? "s" : ""} and excluded from analysis charts.` : "No obvious ID columns were detected.",
  ];
  const memory = Math.max(1, Math.round(JSON.stringify(profile).length / 10));

  const sections = [
    ["Overview", "selected", Grid3X3],
    ["Feature Summary", `${dataset.columns} features`, Table2],
    ["Distributions", `${analysisNumericProfiles.length + analysisCategoricalProfiles.length} charts`, BarChart3],
    ["Correlations", correlationColumns.length > 1 ? "1 matrix" : "0 matrix", LineChart],
    ["Missing Values", `${missingColumns.length} columns`, AlertTriangle],
    ["Outliers", `${analysisNumericProfiles.filter((item) => item.box.outliers > 0).length} features`, Target],
    ["Categorical Insights", `${analysisCategoricalProfiles.length} charts`, PieChart],
    ["AI Insights", `${Math.max(keyFindings.length, profile.warnings?.length ?? 0)} findings`, Sparkles],
  ] as const;

  const metricCards = [
    ["Dataset Shape", `${dataset.rows.toLocaleString()} x ${dataset.columns}`, "Rows x Columns", Grid3X3, "bg-[#F4EDF8]"],
    ["Numeric Columns", numericCount.toLocaleString(), `${Math.round((numericCount / Math.max(1, dataset.columns)) * 100)}% of total`, BarChart3, "bg-[#ECF4FF]"],
    ["Categorical Columns", categoricalCount.toLocaleString(), `${Math.round((categoricalCount / Math.max(1, dataset.columns)) * 100)}% of total`, Table2, "bg-[#EEF5E9]"],
    ["Missing Values", missingTotal.toLocaleString(), "Total missing cells", PieChart, "bg-[#FBF3DF]"],
    ["Duplicate Rows", (profile.duplicates ?? 0).toLocaleString(), (profile.duplicates ?? 0) ? "Duplicates found" : "No duplicates found", FileText, "bg-[#FBEDED]"],
  ] as const;

  return (
    <div className="grid grid-cols-[300px_minmax(0,1fr)] gap-5">
      <aside className="ds-card self-start p-5">
        <h2 className="text-xl font-black">Analysis Sections</h2>
        <p className="mt-2 text-sm leading-6 text-zinc-500">Navigate through different aspects of your exploratory analysis.</p>
        <div className="mt-6 space-y-2">
          {sections.map(([label, count, Icon]) => {
            const isActive = activeSection === label;
            return (
            <button
              key={label}
              className={`grid w-full grid-cols-[24px_minmax(0,1fr)_auto] items-center gap-3 rounded-xl px-4 py-3 text-left text-sm font-bold transition ${
                isActive ? "bg-[#EEF5E9] text-sage" : "hover:bg-stone-50 dark:hover:bg-zinc-800"
              }`}
              onClick={() => setActiveSection(label)}
            >
              <Icon className="h-5 w-5 shrink-0" />
              <span className="min-w-0 whitespace-normal text-[15px] leading-5">{label}</span>
              <span className="flex shrink-0 items-center gap-2 text-xs text-zinc-500">
                <span className="rounded-full bg-white px-2 py-1 dark:bg-zinc-900">{isActive ? "selected" : count}</span>
                {!isActive && <ChevronRight className="h-3 w-3" />}
              </span>
            </button>
            );
          })}
        </div>
        <button className="ds-button-primary mt-8 w-full" onClick={() => setActiveSection("AI Insights")}>
          <Sparkles className="h-4 w-4" />
          Preview Impact
        </button>
        <button
          className="ds-button-secondary mt-3 w-full"
          onClick={() => {
            setActiveSection("Overview");
            setNotice("Insights refreshed from the latest dataset profile.");
          }}
        >
          <RefreshCw className="h-4 w-4" />
          Regenerate Insights
        </button>
      </aside>

      <main className="space-y-5">
        <section className="ds-card p-6">
          <div className="flex items-start justify-between gap-5">
            <div>
              <div className="mb-4 flex items-center gap-4 text-sm text-zinc-500">
                <strong className="text-ink dark:text-zinc-100">{dataset.filename}</strong>
                <span>{dataset.rows.toLocaleString()} rows</span>
                <span>{dataset.columns} columns</span>
                <span>Health {profile.health_score ?? 0}/100</span>
              </div>
              <h1 className="text-3xl font-black">Exploratory Data Analysis</h1>
              <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-300">Understand your dataset through summary statistics, distributions, and relationships.</p>
              {notice && <div className="mt-3 inline-flex rounded-full bg-[#EEF5E9] px-3 py-1 text-xs font-bold text-sage">{notice}</div>}
            </div>
            <div className="flex gap-3">
              <button
                className="ds-button-secondary"
                onClick={() => {
                  const content = `DataStory EDA Report\nDataset: ${dataset.filename}\nRows: ${dataset.rows}\nColumns: ${dataset.columns}\nHealth: ${profile.health_score ?? 0}/100`;
                  downloadTextFile(`${dataset.filename.replace(/\.csv$/i, "")}_eda_report.txt`, content);
                }}
              >
                <Download className="h-4 w-4" />
                Export EDA Report
              </button>
              <button
                className="ds-button-primary"
                onClick={async () => {
                  const message = await shareText(
                    `DataStory insights for ${dataset.filename}`,
                    `DataStory insights for ${dataset.filename}: ${dataset.rows} rows, ${dataset.columns} columns, health ${profile.health_score ?? 0}/100.`,
                  );
                  setNotice(message);
                }}
              >
                <Share2 className="h-4 w-4" />
                Share Insights
              </button>
            </div>
          </div>

          <div className="mt-6 flex gap-8 border-b border-line text-sm font-bold text-zinc-500">
            {["Overview", "Distributions", "Correlations", "Missing Values", "Outliers", "Categorical Insights"].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveSection(tab)}
                className={`border-b-2 pb-3 ${activeSection === tab ? "border-sage text-sage" : "border-transparent hover:text-ink dark:hover:text-zinc-100"}`}
              >
                {tab}
              </button>
            ))}
          </div>

          <div className="mt-5 grid grid-cols-5 gap-4">
            {metricCards.map(([label, value, caption, Icon, tint]) => (
              <div key={label} className="rounded-xl border border-line bg-white p-5 dark:bg-zinc-900">
                <div className="flex items-center gap-4">
                  <span className={`grid h-11 w-11 place-items-center rounded-full ${tint}`}>
                    <Icon className="h-5 w-5 text-sage" />
                  </span>
                  <div>
                    <div className="text-sm text-zinc-500">{label}</div>
                    <div className="mt-1 text-2xl font-black">{value}</div>
                    <div className="text-xs text-zinc-500">{caption}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {activeSection === "Feature Summary" && (
          <section className="ds-card p-5">
            <h3 className="font-black">Feature Summary</h3>
            <div className="mt-4 overflow-hidden rounded-xl border border-line">
              <table className="w-full text-left text-xs">
                <thead className="bg-stone-50 text-zinc-500 dark:bg-zinc-800">
                  <tr>
                    <th className="p-3">Column</th>
                    <th className="p-3">Type</th>
                    <th className="p-3">Missing %</th>
                    <th className="p-3">Unique</th>
                    <th className="p-3">Top Value</th>
                  </tr>
                </thead>
                <tbody>
                  {details.map((row) => (
                    <tr key={String(row.Column)} className="border-t border-line">
                      <td className="p-3 font-bold">{String(row.Column)}</td>
                      <td className="p-3">{String(row.Type)}</td>
                      <td className="p-3">{String(row["Null %"])}%</td>
                      <td className="p-3">{String(row.Unique)}</td>
                      <td className="p-3">{String(row["Top Value"] ?? "-").slice(0, 36)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {activeSection === "Overview" && (
        <section className="grid grid-cols-[1fr_1.9fr_290px] gap-5">
          <div className="ds-card p-5">
            <h3 className="font-black">Column Data Types</h3>
            <Plot
              data={[{ type: "pie", labels: ["Numeric", "Categorical"], values: [numericCount, categoricalCount], hole: 0.58, marker: { colors: ["#4F83E8", "#6FA95F"] } }]}
              layout={{ height: 255, margin: { t: 18, r: 15, b: 15, l: 15 }, paper_bgcolor: "transparent", showlegend: true }}
              config={{ displayModeBar: false, responsive: true }}
              className="w-full"
            />
            <button
              className="mx-auto mt-2 flex w-48 items-center justify-center gap-2 rounded-lg border border-line py-2 text-sm font-bold hover:bg-stone-50 dark:hover:bg-zinc-800"
              onClick={() => setActiveSection("Feature Summary")}
            >
              View All Columns
              <ChevronRight className="h-3 w-3" />
            </button>
          </div>

          <div className="ds-card p-5">
            <div className="flex items-center justify-between">
              <h3 className="font-black">{missingTotal === 0 ? "No Missing Values Found" : "Missing Values by Column"}</h3>
              <button
                className="rounded-lg border border-line px-4 py-2 text-sm font-bold hover:bg-stone-50 dark:hover:bg-zinc-800"
                onClick={() => setActiveSection("Missing Values")}
              >
                View Missing Details
              </button>
            </div>
            {missingTotal === 0 ? (
              <div className="mt-8 grid min-h-[230px] place-items-center rounded-2xl bg-[#F7FAF4] p-6 text-center dark:bg-zinc-900">
                <div>
                  <div className="mx-auto grid h-14 w-14 place-items-center rounded-full bg-[#EEF5E9] text-2xl font-black text-sage">0</div>
                  <p className="mt-4 text-xl font-black">All rows are complete in the current dataset.</p>
                  <p className="mt-2 text-sm text-zinc-500">After cleaning: 0 missing cells. This stage is ready for distribution and relationship analysis.</p>
                </div>
              </div>
            ) : (
              <Plot
                data={[{ type: "bar", x: missingRows.map((row) => row.column), y: missingRows.map((row) => row.count), marker: { color: "#F0B66E" } }]}
                layout={{ height: 285, margin: { t: 25, r: 15, b: 70, l: 55 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { size: 13 } }}
                config={{ displayModeBar: false, responsive: true }}
                className="w-full"
              />
            )}
          </div>

          <aside className="ds-card p-5">
            <h3 className="font-black">Dataset Info</h3>
            {[
              ["Memory Usage", `${memory} KB`],
              ["File Size", `${Math.max(1, Math.round(memory * 0.82))} KB`],
              ["Last Modified", "-"],
              ["Data Source", dataset.filename],
              ["Target Column", targetLabel],
            ].map(([label, value]) => (
              <div key={label} className="flex justify-between gap-4 border-b border-line py-4 text-sm">
                <span className="text-zinc-500">{label}</span>
                <strong className="text-right">{value}</strong>
              </div>
            ))}
            <button
              className="mx-auto mt-5 flex w-44 items-center justify-center gap-2 rounded-lg border border-line py-2 text-sm font-bold hover:bg-stone-50 dark:hover:bg-zinc-800"
              onClick={() => setActiveSection("Feature Summary")}
            >
              View Raw Data
              <ChevronRight className="h-3 w-3" />
            </button>
          </aside>
        </section>
        )}

        {(activeSection === "Overview" || activeSection === "Distributions") && (
        <section className="grid grid-cols-4 gap-5">
          <ChartCard title={`${firstCategorical?.column ?? "Target"} Distribution`} kind="donut" categorical={firstCategorical} />
          <ChartCard title={`${firstNumeric?.column ?? "Numeric"} Distribution`} kind="histogram" numeric={firstNumeric} />
          <ChartCard title={`${secondNumeric?.column ?? "Numeric"} Distribution`} kind="histogram" numeric={secondNumeric} />
          <ChartCard title={`${secondCategorical?.column ?? "Category"} Distribution`} kind="bar" categorical={secondCategorical} />
        </section>
        )}

        {activeSection === "Overview" && (
          <section className="ds-card p-5">
            <div className="flex items-start justify-between gap-5">
              <div>
                <h3 className="font-black">Key Findings</h3>
                <p className="mt-2 text-sm text-zinc-500">DataStory AI highlights the choices that matter before modeling.</p>
              </div>
              <button
                className="ds-button-secondary"
                onClick={() => setActiveSection("AI Insights")}
              >
                Explore All AI Insights
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-4 grid grid-cols-4 gap-3">
              {keyFindings.map((finding) => (
                <div key={finding} className="rounded-xl border border-line bg-stone-50 p-4 text-sm leading-6 dark:bg-zinc-900">
                  {finding}
                </div>
              ))}
            </div>
          </section>
        )}

        {activeSection === "Distributions" && (
          <section className="grid grid-cols-3 gap-5">
            {analysisNumericProfiles.slice(0, 6).map((item) => (
              <ChartCard key={item.column} title={`${item.column} Distribution`} kind="histogram" numeric={item} />
            ))}
            {analysisCategoricalProfiles.slice(0, 6).map((item) => (
              <ChartCard key={item.column} title={`${item.column} Categories`} kind="bar" categorical={item} />
            ))}
          </section>
        )}

        {activeSection === "Missing Values" && (
          <section className="ds-card p-5">
            <h3 className="font-black">Missing Values Detail</h3>
            {missingTotal === 0 ? (
              <div className="mt-4 rounded-2xl border border-line bg-[#F7FAF4] p-8 text-center dark:bg-zinc-900">
                <div className="text-2xl font-black text-sage">No missing values found</div>
                <p className="mt-2 text-sm text-zinc-500">All {dataset.rows.toLocaleString()} rows are complete in the current dataset.</p>
              </div>
            ) : (
              <Plot
                data={[{ type: "bar", x: missingRows.map((row) => row.column), y: missingRows.map((row) => row.count), marker: { color: "#F0B66E" } }]}
                layout={{ height: 360, margin: { t: 25, r: 15, b: 80, l: 55 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { size: 13 } }}
                config={{ displayModeBar: false, responsive: true }}
                className="w-full"
              />
            )}
          </section>
        )}

        {activeSection === "Outliers" && (
          <section className="grid grid-cols-3 gap-5">
            {analysisNumericProfiles.map((item) => (
              <div key={item.column} className="ds-card p-5">
                <h3 className="font-black">{item.column}</h3>
                <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                  {[
                    ["Min", item.box.min],
                    ["Q1", item.box.q1],
                    ["Median", item.box.median],
                    ["Q3", item.box.q3],
                    ["Max", item.box.max],
                    ["IQR Outliers", item.box.outliers],
                  ].map(([label, value]) => (
                    <div key={label} className="rounded-lg bg-stone-50 p-3 dark:bg-zinc-800">
                      <div className="font-bold text-zinc-500">{label}</div>
                      <div className="mt-1 text-lg font-black">{String(value)}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </section>
        )}

        {activeSection === "Categorical Insights" && (
          <section className="grid grid-cols-3 gap-5">
            {analysisCategoricalProfiles.map((item) => (
              <ChartCard key={item.column} title={`${item.column} Categories`} kind="bar" categorical={item} />
            ))}
          </section>
        )}

        {activeSection === "Correlations" && (
          <section className="ds-card p-5">
            <h3 className="font-black">Correlation Matrix</h3>
            <p className="mt-2 text-sm text-zinc-500">Correlation analysis is available for numeric features. Strong relationships can indicate redundancy before modeling.</p>
            {correlationColumns.length < 2 || correlationValues.length < 2 ? (
              <div className="mt-5 rounded-2xl border border-line bg-stone-50 p-8 text-center text-sm text-zinc-500 dark:bg-zinc-900">
                At least two non-ID numeric columns are needed to compute a correlation matrix.
              </div>
            ) : (
              <Plot
                data={[
                  {
                    type: "heatmap",
                    x: correlationColumns,
                    y: correlationColumns,
                    z: correlationValues,
                    zmin: -1,
                    zmax: 1,
                    colorscale: [
                      [0, "#9D4E4E"],
                      [0.5, "#F8F7F2"],
                      [1, "#4F7D3A"],
                    ],
                    hovertemplate: "%{y} vs %{x}<br>Correlation: %{z}<extra></extra>",
                  },
                ]}
                layout={{ height: 470, margin: { t: 30, r: 30, b: 95, l: 105 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { size: 13 } }}
                config={{ displayModeBar: false, responsive: true }}
                className="w-full"
              />
            )}
          </section>
        )}

        {activeSection === "AI Insights" && (
          <section className="grid grid-cols-2 gap-5">
            {[
              ...keyFindings.map((finding) => ({ title: "DataStory Finding", message: finding })),
              ...(profile.warnings ?? []),
            ].map((warning, index) => (
              <div key={`${warning.title}-${index}`} className="ds-card p-5">
                <h3 className="font-black">{warning.title}</h3>
                <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-300">{warning.message}</p>
              </div>
            ))}
          </section>
        )}
      </main>
    </div>
  );
}

function ChartCard({
  title,
  kind,
  numeric,
  categorical,
}: {
  title: string;
  kind: "donut" | "histogram" | "bar";
  numeric?: { histogram: { bins: number[]; counts: number[] } };
  categorical?: { labels: string[]; counts: number[] };
}) {
  const x = numeric?.histogram.bins.slice(0, -1).map((bin, index) => Math.round(((bin + numeric.histogram.bins[index + 1]) / 2) * 100) / 100) ?? [];
  return (
    <div className="ds-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-black">{title}</h3>
        <span className="rounded-lg border border-line px-3 py-1 text-[11px] font-bold text-zinc-500">{kind === "histogram" ? "Histogram" : kind === "bar" ? "Bar" : "Donut"}</span>
      </div>
      <Plot
        data={
          kind === "histogram"
            ? [{ type: "bar", x, y: numeric?.histogram.counts ?? [], marker: { color: "#6FA95F" } }]
            : kind === "donut"
              ? [{ type: "pie", labels: categorical?.labels ?? [], values: categorical?.counts ?? [], hole: 0.58, textinfo: "percent", hoverinfo: "label+value+percent", marker: { colors: ["#6FA95F", "#F06F4F", "#F0B66E", "#4F83E8"] } }]
              : [{ type: "bar", x: categorical?.labels ?? [], y: categorical?.counts ?? [], marker: { color: "#6FA95F" } }]
        }
        layout={{
          height: 225,
          margin: kind === "donut" ? { t: 18, r: 8, b: 18, l: 8 } : { t: 20, r: 10, b: 45, l: 40 },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          showlegend: false,
        }}
        config={{ displayModeBar: false, responsive: true }}
        className="w-full"
      />
    </div>
  );
}
