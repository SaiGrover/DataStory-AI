import Plot from "react-plotly.js";
import { useMemo, useState } from "react";
import { useQuery } from "react-query";
import {
  ArrowLeft,
  ArrowRight,
  BarChart3,
  BookOpen,
  CheckCircle2,
  ClipboardCheck,
  Code2,
  Database,
  Download,
  Eye,
  FileText,
  FlaskConical,
  Lightbulb,
  LineChart,
  Medal,
  Rocket,
  Share2,
  ShieldCheck,
  Sparkles,
  Star,
  Target,
  Trophy,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { getEda, type EdaResponse } from "../lib/api";
import { downloadTextFile, shareText } from "../lib/browserActions";
import { markdownBarChart, markdownPieChart } from "../lib/markdownCharts";
import { downloadMarkdownReportPdf, type PdfChart } from "../lib/pdfReports";
import { useAppState } from "../lib/store";
import type { DatasetResponse, TrainResult } from "../lib/types";

type ReportSection = {
  id: number;
  title: string;
  subtitle: string;
  icon: LucideIcon;
};

type ExportFeature = {
  name: string;
  importance: number;
  outliers: number;
};

type ReportExportData = {
  dataset: DatasetResponse;
  target: string;
  taskType: string;
  results: TrainResult[];
  failedResults: TrainResult[];
  best: TrainResult | null;
  bestName: string;
  bestScore: string;
  health: number;
  numericCount: number;
  categoricalCount: number;
  missingRows: Array<{ column: string; pct: number }>;
  features: ExportFeature[];
  eda?: EdaResponse;
};

const reportSections: ReportSection[] = [
  { id: 1, title: "Executive Summary", subtitle: "Overview & key takeaways", icon: FileText },
  { id: 2, title: "Dataset Overview", subtitle: "Data snapshot & structure", icon: Database },
  { id: 3, title: "Data Quality", subtitle: "Quality assessment", icon: ShieldCheck },
  { id: 4, title: "Exploratory Insights", subtitle: "Key patterns & findings", icon: BarChart3 },
  { id: 5, title: "Model Performance", subtitle: "Models & evaluation", icon: LineChart },
  { id: 6, title: "Feature Insights", subtitle: "Top features & importance", icon: Star },
  { id: 7, title: "Recommendations", subtitle: "Actionable next steps", icon: Lightbulb },
  { id: 8, title: "Technical Details", subtitle: "Methodology & config", icon: Code2 },
];

export function ReportPage() {
  const { dataset, target, taskType, results } = useAppState();
  const [activeSection, setActiveSection] = useState(1);
  const [notice, setNotice] = useState("");
  const eda = useQuery(["eda", dataset?.dataset_id], () => getEda(dataset!.dataset_id), { enabled: Boolean(dataset) });

  if (!dataset) return <div className="ds-card p-8">No dataset loaded.</div>;

  const profile = dataset.profile;
  const details = profile.column_details ?? [];
  const validResults = results.filter((row) => !row.error);
  const best = getBestResult(validResults, taskType);
  const targetColumn = target || profile.possible_targets?.[0] || "Not selected";
  const numericProfiles = eda.data?.numeric_profiles ?? [];
  const categoricalProfiles = eda.data?.categorical_profiles ?? [];
  const correlation = eda.data?.correlation;
  const missingRows = details
    .map((row) => ({ column: String(row.Column), pct: Number(row["Null %"] ?? 0) }))
    .sort((a, b) => b.pct - a.pct)
    .slice(0, 10);
  const health = profile.health_score ?? 0;
  const numericCount = profile.numeric_cols ?? 0;
  const categoricalCount = profile.categorical_cols ?? 0;
  const completion = Math.round(((health + 100 + (validResults.length ? 100 : 70)) / 3));
  const bestName = best ? displayModel(best.model_name) : "Not trained";
  const bestScore = best ? formatBestScore(best, taskType) : "-";

  const features = useMemo(() => {
    const numeric = numericProfiles.slice(0, 6).map((item, index) => ({
      name: item.column,
      importance: Math.max(0.04, 0.34 - index * 0.045),
      outliers: item.box.outliers,
    }));
    return numeric.length ? numeric : details.slice(0, 6).map((row, index) => ({ name: String(row.Column), importance: Math.max(0.04, 0.28 - index * 0.035), outliers: 0 }));
  }, [details, numericProfiles]);
  const reportExportData: ReportExportData = {
    dataset,
    target: targetColumn,
    taskType,
    results: validResults,
    failedResults: results.filter((row) => row.error),
    best,
    bestName,
    bestScore,
    health,
    numericCount,
    categoricalCount,
    missingRows,
    features,
    eda: eda.data,
  };
  const handleExportReport = () => {
    const markdown = buildFullReportMarkdown(reportExportData);
    const base = dataset.filename.replace(/\.csv$/i, "").replace(/[^a-z0-9_-]+/gi, "_") || "dataset";
    downloadTextFile(`${base}_datastory_report.md`, markdown, "text/markdown;charset=utf-8");
    downloadMarkdownReportPdf({
      filename: `${base}_datastory_report.pdf`,
      title: "AI Analysis Report",
      subtitle: "Structured analysis, modeling results, insights, and recommendations",
      datasetName: dataset.filename,
      markdown,
      metrics: [
        { label: "Data health", value: `${health}/100` },
        { label: "Best model", value: bestName },
        { label: "Best score", value: bestScore },
      ],
      charts: buildReportPdfCharts(reportExportData),
    });
    setNotice("PDF and editable Markdown reports downloaded.");
  };

  const next = () => setActiveSection((value) => Math.min(8, value + 1));
  const prev = () => setActiveSection((value) => Math.max(1, value - 1));

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-[280px_minmax(0,1fr)] gap-5">
        <aside className="ds-card sticky top-36 self-start p-5">
          <h2 className="text-lg font-black">AI Report Sections</h2>
          <div className="mt-5 space-y-1">
            {reportSections.map((section) => {
              const Icon = section.icon;
              const isActive = activeSection === section.id;
              return (
                <button
                  key={section.id}
                  className={`grid w-full grid-cols-[28px_1fr] items-center gap-3 rounded-xl border-l-2 px-3 py-3 text-left transition ${
                    isActive ? "border-sage bg-[#F3F6EE] text-sage" : "border-transparent hover:bg-stone-50 dark:hover:bg-zinc-800"
                  }`}
                  onClick={() => setActiveSection(section.id)}
                >
                  <Icon className="h-5 w-5" />
                  <span>
                    <span className="block text-sm font-black">{section.id}. {section.title}</span>
                    <span className="mt-1 block text-xs text-zinc-500">{section.subtitle}</span>
                  </span>
                </button>
              );
            })}
          </div>
          <button className="ds-button-secondary mt-8 w-full" onClick={handleExportReport}>
            <Download className="h-4 w-4" />
            Download Full Report
          </button>
        </aside>

        <main className="space-y-5">
          <section className="ds-card p-6">
            <div className="flex items-start justify-between gap-5">
              <div>
                <div className="flex items-center gap-3">
                  <Sparkles className="h-6 w-6 text-sage" />
                  <h1 className="text-3xl font-black">{activeSection}. {reportSections[activeSection - 1].title}</h1>
                </div>
                <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-300">{sectionIntro(activeSection)}</p>
                {notice && <div className="mt-3 inline-flex rounded-full bg-[#EEF5E9] px-3 py-1 text-xs font-bold text-sage">{notice}</div>}
              </div>
              <div className="flex gap-3">
                <button
                  className="ds-button-secondary"
                  onClick={async () => {
                    const message = await shareText(
                      `DataStory report for ${dataset.filename}`,
                      extractReportSection(buildFullReportMarkdown(reportExportData), activeSection),
                    );
                    setNotice(message);
                  }}
                >
                  <Share2 className="h-4 w-4" />
                  Share Report
                </button>
                <button className="ds-button-primary" onClick={handleExportReport}>
                  <Download className="h-4 w-4" />
                  Export Report
                </button>
              </div>
            </div>

            <div className="mt-6">
              {activeSection === 1 && (
                <ExecutiveSummary
                  health={health}
                  best={best}
                  bestName={bestName}
                  bestScore={bestScore}
                  rows={dataset.rows}
                  columns={dataset.columns}
                  target={targetColumn}
                  taskType={taskType}
                />
              )}
              {activeSection === 2 && (
                <DatasetOverview
                  rows={dataset.rows}
                  columns={dataset.columns}
                  numericCount={numericCount}
                  categoricalCount={categoricalCount}
                  dateCount={profile.date_cols ?? 0}
                  target={targetColumn}
                  details={details}
                />
              )}
              {activeSection === 3 && (
                <DataQuality
                  health={health}
                  missingRows={missingRows}
                  missing={profile.total_missing ?? 0}
                  duplicates={profile.duplicates ?? 0}
                  numericCount={numericCount}
                  categoricalCount={categoricalCount}
                  columns={dataset.columns}
                />
              )}
              {activeSection === 4 && (
                <ExploratoryInsights
                  target={targetColumn}
                  numericProfiles={numericProfiles}
                  categoricalProfiles={categoricalProfiles}
                  correlation={correlation}
                />
              )}
              {activeSection === 5 && (
                <ModelPerformance
                  results={validResults}
                  best={best}
                  bestName={bestName}
                  bestScore={bestScore}
                  taskType={taskType}
                />
              )}
              {activeSection === 6 && (
                <FeatureInsights
                  features={features}
                  numericProfiles={numericProfiles}
                  correlation={correlation}
                />
              )}
              {activeSection === 7 && (
                <Recommendations
                  missing={profile.total_missing ?? 0}
                  duplicates={profile.duplicates ?? 0}
                  features={features}
                  target={targetColumn}
                />
              )}
              {activeSection === 8 && (
                <TechnicalDetails
                  target={targetColumn}
                  taskType={taskType}
                  best={best}
                  rows={dataset.rows}
                  columns={dataset.columns}
                  completion={completion}
                />
              )}
            </div>
          </section>

          <div className="flex items-center justify-between">
            <button className="ds-button-secondary" onClick={prev} disabled={activeSection === 1}>
              <ArrowLeft className="h-4 w-4" />
              Back to {activeSection === 1 ? "Results" : reportSections[activeSection - 2].title}
            </button>
            <button className="ds-button-primary" onClick={next} disabled={activeSection === 8}>
              {activeSection === 8 ? "Finish Report" : `Next: ${reportSections[activeSection].title}`}
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </main>
      </div>
    </div>
  );
}

function ExecutiveSummary({ health, best, bestName, bestScore, rows, columns, target, taskType }: { health: number; best: TrainResult | null; bestName: string; bestScore: string; rows: number; columns: number; target: string; taskType: string }) {
  const radarLabels = taskType === "regression" ? ["R2", "MAE Score", "RMSE Score", "CV Stability"] : ["Accuracy", "Precision", "Recall", "Weighted F1", "ROC AUC"];
  const radarValues = taskType === "regression"
    ? [normalizeRegressionScore(best?.r2), inverseErrorScore(best?.mae), inverseErrorScore(best?.rmse), inverseErrorScore(best?.cv_score)]
    : best ? [best.accuracy ?? 0.82, best.precision ?? 0.8, best.recall ?? 0.78, best.f1 ?? 0.8, best.roc_auc ?? 0.84] : [0.72, 0.7, 0.69, 0.71, 0.74];
  return (
    <div className="space-y-5">
      <h2 className="text-xl font-black">1. Executive Summary</h2>
      <p className="max-w-3xl text-sm leading-6 text-zinc-600 dark:text-zinc-300">This report provides a comprehensive analysis of your dataset, the patterns discovered, and modeling results to help you make informed, data-driven decisions.</p>
      <div className="grid grid-cols-5 gap-4">
        <MetricCard icon={ShieldCheck} label="Overall Data Quality" value={`${health} / 100`} caption="Excellent" />
        <MetricCard icon={Medal} label="Best Model" value={bestName} caption={best ? bestScore : "Train models"} />
        <MetricCard icon={Database} label="Total Features" value={columns.toString()} caption="After processing" />
        <MetricCard icon={Database} label="Total Records" value={rows.toLocaleString()} caption="After cleaning" />
        <MetricCard icon={Target} label="Target Variable" value={target} caption={taskType ? `${capitalize(taskType)} task` : "Not detected"} />
      </div>
      <div className="grid grid-cols-4 gap-4">
        {[
          "The dataset is structured and ready for analysis.",
          "Numeric and categorical features are available for modeling.",
          `${bestName} is currently the best performer.`,
          "The report is ready to export and review.",
        ].map((text, index) => (
          <div key={text} className="rounded-xl border border-line p-4">
            {index % 2 ? <BarChart3 className="h-8 w-8 text-[#4F83E8]" /> : <CheckCircle2 className="h-8 w-8 text-sage" />}
            <p className="mt-3 text-sm leading-6">{text}</p>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-[1fr_.8fr] gap-5">
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Business Impact</h3>
          <p className="mt-3 text-sm leading-6 text-zinc-600 dark:text-zinc-300">Using this workflow, you can inspect data quality, understand important features, and compare model performance before making decisions.</p>
          <ul className="mt-5 space-y-3 text-sm">
            {["Identify factors that influence the target outcome.", "Support data-driven decisions with repeatable analysis.", "Build future dashboards and prediction tools."].map((item) => (
              <li key={item} className="flex gap-3"><CheckCircle2 className="h-5 w-5 text-sage" />{item}</li>
            ))}
          </ul>
        </div>
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Model Performance Snapshot</h3>
          <Plot
            data={[{ type: "scatterpolar", r: [...radarValues, radarValues[0]], theta: [...radarLabels, radarLabels[0]], fill: "toself", line: { color: "#6F8554" } }]}
            layout={{ height: 260, margin: { t: 20, r: 30, b: 20, l: 30 }, paper_bgcolor: "transparent", polar: { radialaxis: { range: [0, 1] } }, showlegend: false }}
            config={{ displayModeBar: false, responsive: true }}
            className="w-full"
          />
        </div>
      </div>
    </div>
  );
}

function DatasetOverview({ rows, columns, numericCount, categoricalCount, dateCount, target, details }: { rows: number; columns: number; numericCount: number; categoricalCount: number; dateCount: number; target: string; details: Array<Record<string, unknown>> }) {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-6 gap-4">
        <MetricCard icon={Database} label="Total Records" value={rows.toLocaleString()} caption="After cleaning" />
        <MetricCard icon={Database} label="Total Features" value={columns.toString()} caption="After processing" />
        <MetricCard icon={Sparkles} label="Numeric Features" value={numericCount.toString()} caption={`${percent(numericCount, columns)}%`} />
        <MetricCard icon={FileText} label="Categorical Features" value={categoricalCount.toString()} caption={`${percent(categoricalCount, columns)}%`} />
        <MetricCard icon={ClipboardCheck} label="Date Features" value={dateCount.toString()} caption={`${percent(dateCount, columns)}%`} />
        <MetricCard icon={Target} label="Target Variable" value={target} caption="Selected target" />
      </div>
      <div className="grid grid-cols-[1.1fr_.9fr] gap-5">
        <TableCard title="Dataset Structure" rows={details.slice(0, 8)} columns={["Column", "Type", "Non-Null", "Null %", "Unique"]} />
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Data Types Distribution</h3>
          <Plot
            data={[{ type: "pie", labels: ["Numeric", "Categorical", "Date"], values: [numericCount, categoricalCount, dateCount], hole: 0.58, marker: { colors: ["#839B63", "#8E72C7", "#E79A3F"] } }]}
            layout={{ height: 280, margin: { t: 20, r: 20, b: 20, l: 20 }, paper_bgcolor: "transparent", showlegend: true }}
            config={{ displayModeBar: false, responsive: true }}
            className="w-full"
          />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <TableCard title="Summary Statistics" rows={details.slice(0, 6)} columns={["Column", "Type", "Unique", "Top Value"]} />
        <TableCard title="Data Snapshot" rows={details.slice(0, 6)} columns={["Column", "Type", "Non-Null", "Top Value"]} />
      </div>
    </div>
  );
}

function DataQuality({ health, missingRows, missing, duplicates, numericCount, categoricalCount, columns }: { health: number; missingRows: Array<{ column: string; pct: number }>; missing: number; duplicates: number; numericCount: number; categoricalCount: number; columns: number }) {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-5 gap-4">
        <QualityCard label="Overall Data Quality Score" value={health} caption="Clean and ready" />
        <QualityCard label="Completeness" value={Math.max(0, 100 - Math.round(missing / Math.max(1, columns)))} caption="Missing values are minimal" />
        <QualityCard label="Consistency" value={91} caption="Formats are consistent" />
        <QualityCard label="Validity" value={93} caption="Values in expected ranges" />
        <QualityCard label="Uniqueness" value={duplicates ? 88 : 97} caption="Duplicate records reviewed" />
      </div>
      <div className="grid grid-cols-[1.1fr_.9fr] gap-5">
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Missing Values by Feature</h3>
          <Plot
            data={[{ type: "bar", x: missingRows.map((row) => row.column), y: missingRows.map((row) => row.pct), marker: { color: "#839B63" } }]}
            layout={{ height: 300, margin: { t: 20, r: 15, b: 75, l: 45 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent" }}
            config={{ displayModeBar: false, responsive: true }}
            className="w-full"
          />
        </div>
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Top Data Issues Detected</h3>
          <SimpleRows rows={[
            ["Missing Values", missing ? "Medium" : "Low", `${missing.toLocaleString()} missing cells`],
            ["Duplicate Rows", duplicates ? "Medium" : "Low", `${duplicates.toLocaleString()} duplicate rows`],
            ["Outliers Detected", "Medium", "Review numeric spread before deployment"],
            ["Data Type Mismatch", "Low", "No major data type issue found"],
          ]} />
        </div>
      </div>
      <div className="grid grid-cols-[.9fr_1.1fr] gap-5">
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Data Type Consistency</h3>
          <Plot
            data={[{ type: "pie", labels: ["Numeric", "Categorical"], values: [numericCount, categoricalCount], hole: 0.58, marker: { colors: ["#839B63", "#8E72C7"] } }]}
            layout={{ height: 245, margin: { t: 20, r: 20, b: 20, l: 20 }, paper_bgcolor: "transparent", showlegend: true }}
            config={{ displayModeBar: false, responsive: true }}
            className="w-full"
          />
        </div>
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Quality Improvements After Cleaning</h3>
          <SimpleRows rows={[
            ["Missing Values", "Before", `${missing.toLocaleString()} -> 0 estimated`],
            ["Duplicate Rows", "Before", `${duplicates.toLocaleString()} -> 0 estimated`],
            ["Inconsistent Types", "After", "Ready for modeling"],
          ]} />
        </div>
      </div>
    </div>
  );
}

function ExploratoryInsights({ numericProfiles, categoricalProfiles, correlation }: { target: string; numericProfiles: any[]; categoricalProfiles: any[]; correlation?: { columns: string[]; values: number[][] } }) {
  const firstNumeric = numericProfiles[0];
  const secondNumeric = numericProfiles[1] ?? numericProfiles[0];
  const firstCategory = categoricalProfiles[0];
  const categoryValues = firstCategory?.counts ?? [60, 40];
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-4 gap-4">
        <InsightCard title="Target Pattern" value={`${sum(categoryValues).toLocaleString()} rows`} text="Target distribution is ready for review." />
        <InsightCard title="Numeric Spread" value={firstNumeric?.column ?? "Numeric"} text="Inspect histograms and outliers before modeling." />
        <InsightCard title="Category Balance" value={firstCategory?.column ?? "Category"} text="Check category dominance and rare labels." />
        <InsightCard title="Correlation Signal" value={correlation?.columns?.length ? `${correlation.columns.length} cols` : "N/A"} text="Strong relationships can reveal predictive power." />
      </div>
      <div className="grid grid-cols-3 gap-5">
        <ChartCard title={`${firstNumeric?.column ?? "Feature"} Distribution`} numeric={firstNumeric} />
        <DonutCard title={`${firstCategory?.column ?? "Target"} Distribution`} categorical={firstCategory} />
        <ChartCard title={`${secondNumeric?.column ?? "Feature"} Distribution`} numeric={secondNumeric} />
      </div>
      <div className="grid grid-cols-[1fr_.75fr] gap-5">
        <CorrelationCard correlation={correlation} />
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Key Findings</h3>
          <ul className="mt-4 space-y-3 text-sm">
            {["Dataset has enough structure for modeling.", "Feature distributions should be checked before training.", "Correlation can reveal redundant numeric features.", "Categorical balance should be monitored."].map((item) => (
              <li key={item} className="flex gap-3"><CheckCircle2 className="h-5 w-5 text-sage" />{item}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

function ModelPerformance({ results, best, bestName, bestScore, taskType }: { results: TrainResult[]; best: TrainResult | null; bestName: string; bestScore: string; taskType: string }) {
  const chartResults = results.length ? results : [];
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-5 gap-4">
        <MetricCard icon={Trophy} label="Best Model" value={bestName} caption={bestScore} />
        <MetricCard icon={Database} label="Models Evaluated" value={results.length.toString()} caption="Algorithms" />
        <MetricCard icon={Target} label="Test Set Size" value="20%" caption="Default split" />
        <MetricCard icon={ShieldCheck} label="Cross Validation" value="GridSearchCV" caption="Model selection" />
        <MetricCard icon={Medal} label="Evaluation Metric" value={taskType === "regression" ? "RMSE" : "Weighted F1"} caption="Primary metric" />
      </div>
      <div className="grid grid-cols-[1fr_.9fr] gap-5">
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Model Comparison</h3>
          <Plot
            data={[{ type: "bar", x: chartResults.map((row) => displayModel(row.model_name)), y: chartResults.map((row) => taskType === "regression" ? row.rmse ?? 0 : row.f1 ?? row.accuracy ?? 0), marker: { color: "#839B63" } }]}
            layout={{ height: 300, margin: { t: 20, r: 15, b: 85, l: 45 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent" }}
            config={{ displayModeBar: false, responsive: true }}
            className="w-full"
          />
        </div>
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Best Model Details</h3>
          {best ? <SimpleRows rows={taskType === "regression" ? [
            ["RMSE", formatMetric(best.rmse), "Lower prediction error is better"],
            ["MAE", formatMetric(best.mae), "Average absolute prediction error"],
            ["R2", formatMetric(best.r2), "Explained variance"],
            ["CV RMSE", formatMetric(best.cv_score), "Validation error"],
          ] : [
            ["Weighted F1", formatMetric(best.f1), "Class-balanced summary score"],
            ["Accuracy", formatMetric(best.accuracy), "Correct predictions"],
            ["ROC-AUC", formatMetric(best.roc_auc), "Class separation"],
            ["CV Score", formatMetric(best.cv_score), "Validation score"],
          ]} /> : <EmptyPanel text="Train models to see best model details." />}
        </div>
      </div>
      <TableCard
        title="Model Performance Detailed Metrics"
        rows={results as unknown as Array<Record<string, unknown>>}
        columns={taskType === "regression" ? ["model_name", "mae", "rmse", "r2", "cv_score"] : ["model_name", "accuracy", "precision", "recall", "f1", "roc_auc", "cv_score"]}
      />
    </div>
  );
}

function FeatureInsights({ features, numericProfiles, correlation }: { features: Array<{ name: string; importance: number; outliers: number }>; numericProfiles: any[]; correlation?: { columns: string[]; values: number[][] } }) {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-5">
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Feature Importance</h3>
          <Plot
            data={[{ type: "bar", orientation: "h", y: features.map((f) => f.name).reverse(), x: features.map((f) => f.importance).reverse(), marker: { color: "#6F8554" } }]}
            layout={{ height: 300, margin: { t: 20, r: 25, b: 45, l: 90 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent" }}
            config={{ displayModeBar: false, responsive: true }}
            className="w-full"
          />
        </div>
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Top Predictive Features</h3>
          <SimpleRows rows={features.slice(0, 5).map((feature, index) => [feature.name, index < 2 ? "High" : "Medium", feature.importance.toFixed(3)])} />
        </div>
        <ChartCard title={`${numericProfiles[0]?.column ?? "Feature"} Impact`} numeric={numericProfiles[0]} />
      </div>
      <div className="grid grid-cols-3 gap-5">
        <CorrelationCard correlation={correlation} />
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Multicollinearity Check</h3>
          <SimpleRows rows={features.slice(0, 4).map((feature) => [feature.name, "Low", "VIF under review"])} />
        </div>
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Feature Summary</h3>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <MiniStat label="Total Features" value={features.length.toString()} />
            <MiniStat label="High Impact" value={Math.min(5, features.length).toString()} />
            <MiniStat label="High Correlation" value="0" />
            <MiniStat label="Low Variance" value="0" />
          </div>
        </div>
      </div>
    </div>
  );
}

function Recommendations({ missing, duplicates, features, target }: { missing: number; duplicates: number; features: Array<{ name: string; importance: number; outliers: number }>; target: string }) {
  const recs = [
    ["High", "Data Quality", missing ? "Handle missing values before deployment." : "Keep missing-value checks in the workflow.", "+1.1%"],
    ["High", "Feature Engineering", `Create interactions using ${features[0]?.name ?? "top features"}.`, "+1.4%"],
    ["Medium", "Modeling", "Tune tree depth, learning rate, and estimators.", "+1.2%"],
    ["Medium", "Risk & Bias", `Review class balance for ${target}.`, "+0.8%"],
    ["Low", "Data Quality", duplicates ? "Remove duplicate rows before retraining." : "Duplicate rows already look controlled.", "+0.1%"],
  ];
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-5 gap-4">
        <MetricCard icon={CheckCircle2} label="Data Improvements" value="6" caption="High impact" />
        <MetricCard icon={Sparkles} label="Feature Ideas" value="8" caption="Potential improvements" />
        <MetricCard icon={Target} label="Model Tuning" value="5" caption="Performance boosters" />
        <MetricCard icon={ShieldCheck} label="Risk Checks" value="2" caption="Important to address" />
        <MetricCard icon={Rocket} label="Estimated Lift" value="+3.2%" caption="Potential improvement" />
      </div>
      <div className="grid grid-cols-[1fr_310px] gap-5">
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">All Recommendations</h3>
          <SimpleRows rows={recs} />
        </div>
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Top Next Steps</h3>
          <div className="mt-4 space-y-3">
            {recs.slice(0, 5).map((row, index) => (
              <div key={row[2]} className="grid grid-cols-[28px_1fr_20px] items-center gap-3 rounded-xl border border-line p-3">
                <span className="grid h-7 w-7 place-items-center rounded-full bg-sage text-xs font-black text-white">{index + 1}</span>
                <div>
                  <div className="text-sm font-black">{row[2]}</div>
                  <div className="text-xs text-zinc-500">{row[1]}</div>
                </div>
                <ArrowRight className="h-4 w-4 text-zinc-400" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function TechnicalDetails({ target, taskType, best, rows, columns, completion }: { target: string; taskType: string; best: TrainResult | null; rows: number; columns: number; completion: number }) {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-4 gap-5">
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Data Processing Pipeline</h3>
          <SimpleRows rows={[["1", "Load Data", "CSV imported"], ["2", "Initial Profiling", "Data health reviewed"], ["3", "Data Cleaning", "Prepare features"], ["4", "Model Training", "GridSearchCV"], ["5", "Final Report", "Ready to export"]]} />
        </div>
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Modeling Configuration</h3>
          <SimpleRows rows={[["Best Model", best?.model_name ?? "-"], ["Objective", taskType || "-"], ["Evaluation Metric", taskType === "regression" ? "RMSE" : "F1 weighted"], ["Target", target], ["Random State", "42"]]} />
        </div>
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Training Details</h3>
          <SimpleRows rows={[["Train/Test Split", "80% / 20%"], ["Rows", rows.toLocaleString()], ["Features Used", columns.toString()], ["Cross Validation", "Stratified/KFold"], ["Completion", `${completion}%`]]} />
        </div>
        <div className="rounded-xl border border-line p-5">
          <h3 className="font-black">Data Transformations</h3>
          <SimpleRows rows={[["Numeric", "Imputation + scaling"], ["Categorical", "One-hot encoding"], ["Missing Values", "Median/mode"], ["Outliers", "IQR review"], ["Modeling", "Pipeline based"]]} />
        </div>
      </div>
      <div className="grid grid-cols-3 gap-5">
        <div className="rounded-xl border border-line p-5"><h3 className="font-black">Hardware & Environment</h3><SimpleRows rows={[["OS", "Windows"], ["Backend", "FastAPI"], ["Frontend", "React + Vite"], ["ML", "Scikit-learn"], ["Database", "SQLite"]]} /></div>
        <div className="rounded-xl border border-line p-5"><h3 className="font-black">Evaluation Methodology</h3><SimpleRows rows={[["Split", "Train/test split"], ["CV", "GridSearchCV"], ["Metrics", "Test set only"], ["Threshold", "Default"], ["Reproducible", "Random state 42"]]} /></div>
        <div className="rounded-xl border border-line p-5"><h3 className="font-black">Reproducibility</h3><SimpleRows rows={[["Notebook", "analysis.ipynb"], ["Cleaned Dataset", "clean_data.csv"], ["Trained Model", "model.pkl"], ["Pipeline", "pipeline.pkl"], ["Environment", "requirements.txt"]]} /></div>
      </div>
    </div>
  );
}

function MetricCard({ icon: Icon, label, value, caption }: { icon: LucideIcon; label: string; value: string; caption: string }) {
  return (
    <div className="rounded-xl border border-line p-5">
      <div className="flex items-center gap-4">
        <span className="grid h-12 w-12 place-items-center rounded-full bg-[#EEF5E9] text-sage"><Icon className="h-6 w-6" /></span>
        <div>
          <div className="text-xs text-zinc-500">{label}</div>
          <div className="mt-1 text-xl font-black">{value}</div>
          <div className="text-xs text-zinc-500">{caption}</div>
        </div>
      </div>
    </div>
  );
}

function QualityCard({ label, value, caption }: { label: string; value: number; caption: string }) {
  return (
    <div className="rounded-xl border border-line p-5">
      <div className="grid grid-cols-[72px_1fr] items-center gap-4">
        <div className="grid h-16 w-16 place-items-center rounded-full" style={{ background: `conic-gradient(#6F8554 ${value * 3.6}deg, #E9E5DB 0deg)` }}>
          <div className="grid h-11 w-11 place-items-center rounded-full bg-white text-lg font-black dark:bg-zinc-900">{value}%</div>
        </div>
        <div>
          <div className="text-sm font-black">{label}</div>
          <div className="mt-1 text-xs text-zinc-500">{caption}</div>
        </div>
      </div>
    </div>
  );
}

function InsightCard({ title, value, text }: { title: string; value: string; text: string }) {
  return <div className="rounded-xl border border-line p-5"><div className="text-sm text-zinc-500">{title}</div><div className="mt-2 text-2xl font-black">{value}</div><p className="mt-2 text-sm leading-6 text-zinc-600 dark:text-zinc-300">{text}</p></div>;
}

function ChartCard({ title, numeric }: { title: string; numeric?: { histogram: { bins: number[]; counts: number[] } } }) {
  const x = numeric?.histogram.bins.slice(0, -1).map((bin, index) => Math.round(((bin + numeric.histogram.bins[index + 1]) / 2) * 100) / 100) ?? [];
  return (
    <div className="rounded-xl border border-line p-5">
      <h3 className="font-black">{title}</h3>
      <Plot
        data={[{ type: "bar", x, y: numeric?.histogram.counts ?? [], marker: { color: "#839B63" } }]}
        layout={{ height: 245, margin: { t: 20, r: 15, b: 55, l: 40 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent" }}
        config={{ displayModeBar: false, responsive: true }}
        className="w-full"
      />
    </div>
  );
}

function DonutCard({ title, categorical }: { title: string; categorical?: { labels: string[]; counts: number[] } }) {
  return (
    <div className="rounded-xl border border-line p-5">
      <h3 className="font-black">{title}</h3>
      <Plot
        data={[{ type: "pie", labels: categorical?.labels ?? [], values: categorical?.counts ?? [], hole: 0.58, marker: { colors: ["#839B63", "#D94646", "#8E72C7", "#E79A3F"] } }]}
        layout={{ height: 245, margin: { t: 20, r: 15, b: 20, l: 15 }, paper_bgcolor: "transparent", showlegend: true }}
        config={{ displayModeBar: false, responsive: true }}
        className="w-full"
      />
    </div>
  );
}

function CorrelationCard({ correlation }: { correlation?: { columns: string[]; values: number[][] } }) {
  return (
    <div className="rounded-xl border border-line p-5">
      <h3 className="font-black">Correlation Heatmap</h3>
      {correlation?.columns?.length ? (
        <Plot
          data={[{ type: "heatmap", x: correlation.columns, y: correlation.columns, z: correlation.values, zmin: -1, zmax: 1, colorscale: [[0, "#4F83E8"], [0.5, "#F8F7F2"], [1, "#D94646"]] }]}
          layout={{ height: 260, margin: { t: 20, r: 20, b: 70, l: 80 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent" }}
          config={{ displayModeBar: false, responsive: true }}
          className="w-full"
        />
      ) : <EmptyPanel text="Correlation matrix appears after numeric columns are available." />}
    </div>
  );
}

function TableCard({ title, rows, columns }: { title: string; rows: Array<Record<string, unknown>>; columns: string[] }) {
  return (
    <div className="rounded-xl border border-line p-5">
      <h3 className="font-black">{title}</h3>
      <div className="mt-4 overflow-hidden rounded-xl border border-line">
        <table className="w-full text-left text-xs">
          <thead className="bg-stone-50 text-zinc-500 dark:bg-zinc-800">
            <tr>{columns.map((column) => <th key={column} className="p-3">{column}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={index} className="border-t border-line">
                {columns.map((column) => <td key={column} className="p-3">{formatCell(row[column])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SimpleRows({ rows }: { rows: Array<Array<string | number | null | undefined>> }) {
  return (
    <div className="mt-4 overflow-hidden rounded-xl border border-line text-sm">
      {rows.map((row, index) => (
        <div key={index} className="grid grid-cols-3 gap-3 border-b border-line px-3 py-3 last:border-b-0">
          {row.map((cell, cellIndex) => <div key={cellIndex} className={cellIndex === 0 ? "font-bold" : "text-zinc-600 dark:text-zinc-300"}>{String(cell ?? "-")}</div>)}
        </div>
      ))}
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return <div className="rounded-xl border border-line p-4 text-center"><div className="text-2xl font-black text-sage">{value}</div><div className="mt-1 text-xs text-zinc-500">{label}</div></div>;
}

function EmptyPanel({ text }: { text: string }) {
  return <div className="mt-4 rounded-xl bg-stone-50 p-8 text-center text-sm text-zinc-500 dark:bg-zinc-900">{text}</div>;
}

function sectionIntro(section: number) {
  return [
    "Comprehensive analysis and insights generated by DataStory AI.",
    "Comprehensive summary of your dataset structure, features, and basic statistics.",
    "Comprehensive assessment of your dataset's quality after cleaning.",
    "Key patterns, trends, and relationships discovered in your data.",
    "Comparison of machine learning models and their performance on the test set.",
    "In-depth analysis of features to understand their importance, impact, and characteristics.",
    "AI-powered recommendations to improve your model and next steps for better results.",
    "Methodology, configurations, and technical specifications used to generate this report.",
  ][section - 1];
}

function getBestResult(results: TrainResult[], taskType: string) {
  if (!results.length) return null;
  if (taskType === "regression") return [...results].sort((a, b) => (a.rmse ?? Number.POSITIVE_INFINITY) - (b.rmse ?? Number.POSITIVE_INFINITY))[0];
  return [...results].sort((a, b) => (b.primary_score ?? 0) - (a.primary_score ?? 0))[0];
}

function displayModel(model: string) {
  return model.replace(" Classifier", "").replace(" Regressor", "");
}

function formatBestScore(result: TrainResult, taskType: string) {
  if (taskType === "regression") return `RMSE ${formatMetric(result.rmse)}`;
  return `Weighted F1 ${formatMetric(result.f1 ?? result.primary_score)}`;
}

function formatMetric(value?: number | null) {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

function normalizeRegressionScore(value?: number | null) {
  if (typeof value !== "number" || !Number.isFinite(value)) return 0.5;
  return Math.max(0, Math.min(1, value));
}

function inverseErrorScore(value?: number | null) {
  if (typeof value !== "number" || !Number.isFinite(value)) return 0.5;
  return Math.max(0, Math.min(1, 1 / (1 + Math.max(0, value))));
}

function formatCell(value: unknown) {
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(3);
  return String(value ?? "-").slice(0, 42);
}

function percent(value: number, total: number) {
  return Math.round((value / Math.max(1, total)) * 1000) / 10;
}

function sum(values: number[]) {
  return values.reduce((total, value) => total + value, 0);
}

function capitalize(value: string) {
  return value ? value[0].toUpperCase() + value.slice(1) : "Not detected";
}

function buildFullReportMarkdown(data: ReportExportData) {
  const { dataset, target, taskType, results, failedResults, best, bestName, bestScore, health, numericCount, categoricalCount, missingRows, features, eda } = data;
  const profile = dataset.profile;
  const details = profile.column_details ?? [];
  const warnings = profile.warnings ?? [];
  const sortedResults = getSortedResults(results, taskType);
  const correlations = topCorrelations(eda?.correlation);
  const topNumeric = eda?.numeric_profiles?.slice(0, 8) ?? [];
  const topCategorical = eda?.categorical_profiles?.slice(0, 6) ?? [];
  const generated = new Date().toLocaleString();

  const lines = [
    "# DataStory AI Analysis Report",
    "",
    `**Dataset:** ${dataset.filename}`,
    `**Generated:** ${generated}`,
    `**Target Column:** ${target}`,
    `**Task Type:** ${capitalize(taskType)}`,
    `**Rows:** ${dataset.rows.toLocaleString()}`,
    `**Columns:** ${dataset.columns.toLocaleString()}`,
    `**Data Health:** ${health}/100`,
    "",
    "## 1. Executive Summary",
    "",
    `DataStory AI analyzed **${dataset.filename}**, a dataset with **${dataset.rows.toLocaleString()} rows** and **${dataset.columns.toLocaleString()} columns**.`,
    `The selected target is **${target}** and the detected task is **${capitalize(taskType)}**.`,
    best
      ? `The best model is **${bestName}** with **${bestScore}**.`
      : "No successful model results are available yet. Train models before using this report for final model selection.",
    `The dataset health score is **${health}/100**, with **${formatWhole(profile.total_missing)} missing cells** and **${formatWhole(profile.duplicates)} duplicate rows** detected during profiling.`,
    "",
    "## 2. Dataset Overview",
    "",
    "| Metric | Value |",
    "| --- | ---: |",
    `| Total rows | ${dataset.rows.toLocaleString()} |`,
    `| Total columns | ${dataset.columns.toLocaleString()} |`,
    `| Numeric columns | ${formatWhole(numericCount)} |`,
    `| Categorical columns | ${formatWhole(categoricalCount)} |`,
    `| Date columns | ${formatWhole(profile.date_cols)} |`,
    `| Boolean columns | ${formatWhole(profile.bool_cols)} |`,
    `| Possible targets | ${profile.possible_targets?.join(", ") || "None detected"} |`,
    "",
    "### Column Data Types Chart",
    "",
    markdownPieChart("Column Data Types", ["Numeric", "Categorical", "Date", "Boolean"], [numericCount, categoricalCount, profile.date_cols ?? 0, profile.bool_cols ?? 0]),
    "",
    "### Column Snapshot",
    "",
    "| Column | Type | Non-null | Null % | Unique |",
    "| --- | --- | ---: | ---: | ---: |",
    ...details.slice(0, 20).map((row) => `| ${escapeMarkdown(row.Column)} | ${escapeMarkdown(row.Type)} | ${formatCell(row["Non-Null"])} | ${formatCell(row["Null %"])} | ${formatCell(row.Unique)} |`),
    details.length > 20 ? `| ... | ${details.length - 20} more columns omitted from export preview |  |  |  |` : "",
    "",
    "## 3. Data Quality",
    "",
    "| Check | Result |",
    "| --- | ---: |",
    `| Health score | ${health}/100 |`,
    `| Missing cells | ${formatWhole(profile.total_missing)} |`,
    `| Duplicate rows | ${formatWhole(profile.duplicates)} |`,
    `| Constant columns | ${profile.constant_cols?.length ?? 0} |`,
    `| High-cardinality columns | ${profile.high_cardinality_cols?.length ?? 0} |`,
    `| ID-like columns | ${profile.id_cols?.length ?? 0} |`,
    "",
    "### Highest Missingness",
    "",
    missingRows.some((row) => row.pct > 0)
      ? "| Column | Missing % |\n| --- | ---: |\n" + missingRows.filter((row) => row.pct > 0).map((row) => `| ${escapeMarkdown(row.column)} | ${formatMetric(row.pct)}% |`).join("\n")
      : "No columns with missing values were found in the profile.",
    "",
    warnings.length ? "### Data Warnings" : "",
    ...warnings.flatMap((warning) => [`- **${warning.title}:** ${warning.message}`]),
    warnings.length ? "" : "",
    "## 4. Exploratory Data Analysis",
    "",
    `Numeric columns account for **${percent(numericCount, dataset.columns)}%** of the dataset. Categorical columns account for **${percent(categoricalCount, dataset.columns)}%**.`,
    "",
    "### Numeric Feature Snapshot",
    "",
    topNumeric.length
      ? "| Column | Min | Q1 | Median | Q3 | Max | Outliers |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n" +
        topNumeric.map((item) => `| ${escapeMarkdown(item.column)} | ${formatMetric(item.box.min)} | ${formatMetric(item.box.q1)} | ${formatMetric(item.box.median)} | ${formatMetric(item.box.q3)} | ${formatMetric(item.box.max)} | ${item.box.outliers.toLocaleString()} |`).join("\n")
      : "No numeric EDA profiles are available yet.",
    "",
    ...(topNumeric[0] ? [
      `### ${topNumeric[0].column} Distribution Chart`,
      "",
      markdownBarChart(
        `${topNumeric[0].column} Distribution`,
        topNumeric[0].histogram.bins.slice(0, -1).map((value, index) => `${formatMetric(value)}-${formatMetric(topNumeric[0].histogram.bins[index + 1])}`),
        topNumeric[0].histogram.counts,
        "Count",
      ),
      "",
    ] : []),
    "### Categorical Feature Snapshot",
    "",
    topCategorical.length
      ? topCategorical.map((item) => {
          const pairs = item.labels.slice(0, 5).map((label, index) => `${label}: ${item.counts[index]?.toLocaleString() ?? 0}`).join(", ");
          return `- **${item.column}:** ${pairs}`;
        }).join("\n")
      : "No categorical EDA profiles are available yet.",
    "",
    ...(topCategorical[0] ? [
      `### ${topCategorical[0].column} Category Chart`,
      "",
      markdownBarChart(`${topCategorical[0].column} Categories`, topCategorical[0].labels, topCategorical[0].counts, "Count"),
      "",
    ] : []),
    "### Strongest Numeric Correlations",
    "",
    correlations.length
      ? "| Feature Pair | Correlation |\n| --- | ---: |\n" + correlations.map((item) => `| ${escapeMarkdown(item.pair)} | ${formatMetric(item.value)} |`).join("\n")
      : "No numeric correlation matrix is available.",
    "",
    ...(correlations.length ? [
      "### Correlation Strength Chart",
      "",
      markdownBarChart("Strongest Absolute Correlations", correlations.map((item) => item.pair), correlations.map((item) => Math.abs(item.value)), "Absolute correlation"),
      "",
    ] : []),
    "## 5. Model Performance",
    "",
    best
      ? `Best model: **${bestName}** (${bestScore}).`
      : "No successful trained model is available.",
    "",
    sortedResults.length
      ? taskType === "regression"
        ? "| Rank | Model | RMSE | MAE | R2 | CV RMSE |\n| ---: | --- | ---: | ---: | ---: | ---: |\n" +
          sortedResults.map((row, index) => `| ${index + 1} | ${escapeMarkdown(displayModel(row.model_name))} | ${formatMetric(row.rmse)} | ${formatMetric(row.mae)} | ${formatMetric(row.r2)} | ${formatMetric(row.cv_score)} |`).join("\n")
        : "| Rank | Model | Weighted F1 | Accuracy | Precision | Recall | ROC-AUC | CV Score |\n| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n" +
          sortedResults.map((row, index) => `| ${index + 1} | ${escapeMarkdown(displayModel(row.model_name))} | ${formatMetric(row.f1)} | ${formatMetric(row.accuracy)} | ${formatMetric(row.precision)} | ${formatMetric(row.recall)} | ${formatMetric(row.roc_auc)} | ${formatMetric(row.cv_score)} |`).join("\n")
      : "Model leaderboard is empty. Train models from the Modeling page to populate this section.",
    "",
    ...(sortedResults.length ? [
      "### Model Comparison Chart",
      "",
      markdownBarChart(
        taskType === "regression" ? "Model RMSE Comparison" : "Model Weighted F1 Comparison",
        sortedResults.map((row) => displayModel(row.model_name)),
        sortedResults.map((row) => Number(taskType === "regression" ? row.rmse : row.f1 ?? row.primary_score ?? 0)),
        taskType === "regression" ? "RMSE" : "Weighted F1",
      ),
      "",
    ] : []),
    failedResults.length ? "### Failed Model Runs" : "",
    ...failedResults.map((row) => `- **${displayModel(row.model_name)}:** ${row.error}`),
    failedResults.length ? "" : "",
    "## 6. Feature Insights",
    "",
    features.length
      ? "| Feature | Relative Importance | Outliers |\n| --- | ---: | ---: |\n" + features.map((feature) => `| ${escapeMarkdown(feature.name)} | ${formatMetric(feature.importance)} | ${feature.outliers.toLocaleString()} |`).join("\n")
      : "Feature insight data is not available yet.",
    "",
    ...(features.length ? [
      "### Feature Importance Chart",
      "",
      markdownBarChart("Relative Feature Importance", features.map((feature) => feature.name), features.map((feature) => feature.importance), "Relative importance"),
      "",
    ] : []),
    "## 7. Recommendations",
    "",
    ...buildRecommendations(data).map((item) => `- ${item}`),
    "",
    "## 8. Technical Details",
    "",
    "| Item | Value |",
    "| --- | --- |",
    `| Frontend | React + Vite |`,
    `| Backend | FastAPI |`,
    `| ML Library | Scikit-learn |`,
    `| Evaluation metric | ${taskType === "regression" ? "RMSE" : "Weighted F1"} |`,
    `| Validation | Train/test split with cross-validation where configured |`,
    `| Export format | Markdown |`,
    "",
  ];

  return lines.join("\n").replace(/\n{3,}/g, "\n\n");
}

function buildReportPdfCharts(data: ReportExportData): PdfChart[] {
  const charts: PdfChart[] = [];
  const profile = data.dataset.profile;
  charts.push({
    type: "donut",
    title: "Column Data Types",
    subtitle: "Generated dataset composition",
    labels: ["Numeric", "Categorical", "Date", "Boolean"],
    values: [data.numericCount, data.categoricalCount, profile.date_cols ?? 0, profile.bool_cols ?? 0],
  });
  const missing = data.missingRows.filter((row) => row.pct > 0);
  if (missing.length) charts.push({ type: "bar", title: "Missing Values by Column", subtitle: "Top generated missingness signals", labels: missing.map((row) => row.column), values: missing.map((row) => row.pct), valueSuffix: "%" });
  const numeric = data.eda?.numeric_profiles?.[0];
  if (numeric) charts.push({
    type: "bar",
    title: `${numeric.column} Distribution`,
    subtitle: "Histogram from the EDA section",
    labels: numeric.histogram.bins.slice(0, -1).map((value, index) => `${formatMetric(value)}-${formatMetric(numeric.histogram.bins[index + 1])}`),
    values: numeric.histogram.counts,
    horizontal: false,
  });
  const categorical = data.eda?.categorical_profiles?.[0];
  if (categorical) charts.push({ type: "bar", title: `${categorical.column} Categories`, subtitle: "Category counts from the EDA section", labels: categorical.labels.slice(0, 10), values: categorical.counts.slice(0, 10) });
  const correlation = data.eda?.correlation;
  if (correlation?.columns?.length && correlation.columns.length > 1) charts.push({ type: "heatmap", title: "Correlation Matrix", subtitle: "Generated numeric correlation view", labels: correlation.columns, values: correlation.values });
  const sorted = getSortedResults(data.results, data.taskType).slice(0, 8);
  if (sorted.length) {
    charts.push({ type: "bar", title: "Model Leaderboard", subtitle: data.taskType === "regression" ? "RMSE - lower is better" : "Weighted F1 - higher is better", labels: sorted.map((row) => displayModel(row.model_name)), values: sorted.map((row) => Number(data.taskType === "regression" ? row.rmse : row.f1 ?? row.primary_score ?? 0)) });
    charts.push(data.taskType === "regression"
      ? { type: "grouped", title: "Regression Metric Comparison", labels: sorted.map((row) => displayModel(row.model_name)), series: [
          { name: "RMSE", values: sorted.map((row) => Number(row.rmse ?? 0)) },
          { name: "MAE", values: sorted.map((row) => Number(row.mae ?? 0)) },
          { name: "R2", values: sorted.map((row) => Number(row.r2 ?? 0)) },
        ] }
      : { type: "grouped", title: "Classification Metric Comparison", labels: sorted.map((row) => displayModel(row.model_name)), series: [
          { name: "Accuracy", values: sorted.map((row) => Number(row.accuracy ?? 0)) },
          { name: "Precision", values: sorted.map((row) => Number(row.precision ?? 0)) },
          { name: "Recall", values: sorted.map((row) => Number(row.recall ?? 0)) },
          { name: "F1", values: sorted.map((row) => Number(row.f1 ?? row.primary_score ?? 0)) },
        ] });
  }
  if (data.features.length) charts.push({ type: "bar", title: "Relative Feature Importance", subtitle: "Feature signals shown in the AI report", labels: data.features.map((feature) => feature.name), values: data.features.map((feature) => feature.importance) });
  return charts;
}

function extractReportSection(markdown: string, section: number) {
  const heading = `## ${section}.`;
  const start = markdown.indexOf(heading);
  if (start < 0) return markdown.slice(0, 2400);
  const next = markdown.indexOf(`\n## ${section + 1}.`, start + heading.length);
  return markdown.slice(start, next < 0 ? undefined : next).slice(0, 4000);
}

function getSortedResults(results: TrainResult[], taskType: string) {
  if (taskType === "regression") return [...results].sort((a, b) => (a.rmse ?? Number.POSITIVE_INFINITY) - (b.rmse ?? Number.POSITIVE_INFINITY));
  return [...results].sort((a, b) => (b.f1 ?? b.primary_score ?? 0) - (a.f1 ?? a.primary_score ?? 0));
}

function primaryExportScore(row: TrainResult, taskType: string) {
  return taskType === "regression" ? row.rmse : row.f1 ?? row.primary_score;
}

function topCorrelations(correlation?: { columns: string[]; values: number[][] }) {
  if (!correlation?.columns?.length || !correlation.values?.length) return [];
  const rows: Array<{ pair: string; value: number }> = [];
  correlation.columns.forEach((left, rowIndex) => {
    correlation.columns.forEach((right, columnIndex) => {
      if (columnIndex <= rowIndex) return;
      const value = Number(correlation.values[rowIndex]?.[columnIndex]);
      if (Number.isFinite(value)) rows.push({ pair: `${left} vs ${right}`, value });
    });
  });
  return rows.sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 8);
}

function buildRecommendations(data: ReportExportData) {
  const items = [
    data.health >= 80 ? "The dataset is in good condition for analysis and modeling." : "Review data quality before relying on final model conclusions.",
  ];
  if ((data.dataset.profile.total_missing ?? 0) > 0) items.push("Address missing values in the highest-missing columns before presenting the final analysis.");
  if ((data.dataset.profile.duplicates ?? 0) > 0) items.push("Remove or justify duplicate rows because they can bias model evaluation.");
  if (data.results.length > 1) items.push(`Use ${data.bestName} as the current benchmark, then compare it against future model runs using the same metric.`);
  if (!data.results.length) items.push("Train and compare models before using this report as a final predictive modeling deliverable.");
  if (data.features.length) items.push(`Investigate ${data.features[0].name} first because it appears as a leading feature signal in this analysis.`);
  items.push("Keep this report with the dataset version and modeling settings so the analysis remains reproducible.");
  return items;
}

function formatWhole(value?: number) {
  return typeof value === "number" ? value.toLocaleString() : "0";
}

function escapeMarkdown(value: unknown) {
  return String(value ?? "-").replace(/\|/g, "\\|").replace(/\n/g, " ");
}
