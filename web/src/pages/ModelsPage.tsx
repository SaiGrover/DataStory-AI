import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "react-query";
import { useNavigate } from "react-router-dom";
import {
  AlertTriangle,
  BarChart3,
  Check,
  ChevronRight,
  Clock3,
  Download,
  FlaskConical,
  LineChart,
  RefreshCw,
  Share2,
  SlidersHorizontal,
  Target,
  Trophy,
} from "lucide-react";
import { listModels, selectTarget, trainModels, type ImbalanceInfo } from "../lib/api";
import { shareText } from "../lib/browserActions";
import { markdownBarChart } from "../lib/markdownCharts";
import { downloadModelingReportPdf } from "../lib/pdfReports";
import { useAppState } from "../lib/store";
import type { TrainResult } from "../lib/types";

const MODEL_META: Record<string, { family: string; short: string }> = {
  "Logistic Regression": { family: "Linear Model", short: "Logistic Regression" },
  "Decision Tree Classifier": { family: "Tree Based", short: "Decision Tree" },
  "Random Forest Classifier": { family: "Ensemble", short: "Random Forest" },
  "Support Vector Machine": { family: "Margin Based", short: "SVM" },
  "K-Nearest Neighbors": { family: "Instance Based", short: "K Nearest Neighbors" },
  "Naive Bayes": { family: "Probabilistic", short: "Naive Bayes" },
  "Gradient Boosting Classifier": { family: "Ensemble", short: "Gradient Boosting" },
  "Extra Trees Classifier": { family: "Ensemble", short: "Extra Trees" },
  "Linear Regression": { family: "Linear Model", short: "Linear Regression" },
  "Ridge Regression": { family: "Linear Model", short: "Ridge Regression" },
  "Decision Tree Regressor": { family: "Tree Based", short: "Decision Tree" },
  "Random Forest Regressor": { family: "Ensemble", short: "Random Forest" },
  "Gradient Boosting Regressor": { family: "Ensemble", short: "Gradient Boosting" },
  "Extra Trees Regressor": { family: "Ensemble", short: "Extra Trees" },
  "Support Vector Regressor": { family: "Margin Based", short: "SVR" },
};

export function ModelsPage() {
  const { dataset, target, setTarget, taskType, setTaskType, results, setResults } = useAppState();
  const navigate = useNavigate();
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [testSize, setTestSize] = useState("0.2");
  const [cvFolds, setCvFolds] = useState("5");
  const [imbalanceInfo, setImbalanceInfo] = useState<ImbalanceInfo | null>(null);
  const [imbalanceStrategy, setImbalanceStrategy] = useState<"none" | "smote" | "class_weight">("none");
  const [notice, setNotice] = useState("");

  const recommendedTarget = dataset?.profile.possible_targets?.[0] ?? "";

  useEffect(() => {
    if (dataset && !target && recommendedTarget) setTarget(recommendedTarget);
  }, [dataset, recommendedTarget, setTarget, target]);

  const detect = useMutation(() => selectTarget(dataset!.dataset_id, target), {
    onSuccess: (data) => {
      setTaskType(data.task_type);
      setImbalanceInfo(data.imbalance_info ?? null);
      const isImbalanced = Boolean(data.imbalance_info?.imbalanced ?? data.imbalance_info?.is_imbalanced);
      setImbalanceStrategy(data.task_type === "classification" && isImbalanced ? "smote" : "none");
      setNotice(
        data.task_type === "classification" && isImbalanced
          ? "Target detected as classification with imbalance. SMOTE is available and selected."
          : `Target detected as ${data.task_type}. Choose models and start training.`,
      );
    },
  });

  const models = useQuery(["models", taskType], () => listModels(taskType || "classification"), {
    enabled: Boolean(taskType),
  });

  useEffect(() => {
    const available = models.data?.models ?? [];
    if (available.length && selectedModels.length === 0) {
      setSelectedModels(available.slice(0, Math.min(6, available.length)));
    }
  }, [models.data?.models, selectedModels.length]);

  const train = useMutation(
    () =>
      trainModels(dataset!.dataset_id, target, taskType, selectedModels, {
        testSize: Number(testSize),
        cvFolds: Number(cvFolds),
        imbalanceStrategy,
      }),
    {
      onSuccess: (data) => {
        setResults(data.results);
        setNotice("Training complete. Results are ready for comparison.");
      },
    },
  );

  if (!dataset) return <div className="ds-card p-8">No dataset loaded.</div>;

  const availableModels = taskType ? models.data?.models ?? [] : [];
  const validResults = results.filter((row) => !row.error);
  const best = validResults.length ? [...validResults].sort(compareResults(taskType))[0] : null;
  const selectedRows = dataset.rows - Math.round(dataset.rows * Number(testSize));
  const testRows = dataset.rows - selectedRows;

  const metricCards = [
    ["Best Model", best ? displayModel(best.model_name) : "-", best ? primaryScoreLabel(best, taskType) : "Train to compare", Trophy],
    ["Models Selected", selectedModels.length.toString(), "Ready to train", FlaskConical],
    ["Best Score", best ? formatPrimaryScore(best, taskType) : "-", best?.model_name ? displayModel(best.model_name) : "No run yet", Target],
    ["Training Status", train.isLoading ? "Running" : validResults.length ? "Complete" : "Not started", train.isLoading ? "GridSearchCV in progress" : "Local backend", Clock3],
    ["Data Split", `${selectedRows} / ${testRows}`, "Train / Test", BarChart3],
  ] as const;

  return (
    <div>
      <main className="space-y-5">
        <section className="ds-card p-6">
          <div className="flex items-start justify-between gap-5">
            <div className="flex gap-4">
              <span className="grid h-14 w-14 place-items-center rounded-full bg-[#EEF5E9] text-sage"><FlaskConical className="h-7 w-7" /></span>
              <div>
                <div className="mb-4 flex items-center gap-4 text-sm text-zinc-500">
                  <strong className="text-ink dark:text-zinc-100">{dataset.filename}</strong>
                  <span>{dataset.rows.toLocaleString()} rows</span>
                  <span>{dataset.columns} columns</span>
                  <span>Health {dataset.profile.health_score ?? 0}/100</span>
                </div>
                <h1 className="text-3xl font-black">Modeling</h1>
                <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-300">Choose a target column, select models, train them, and compare the best performer.</p>
                {notice && <div className="mt-3 inline-flex rounded-full bg-[#EEF5E9] px-3 py-1 text-xs font-bold text-sage">{notice}</div>}
              </div>
            </div>
            <div className="flex gap-3">
              <button
                className="ds-button-secondary"
                onClick={() => {
                  downloadModelingReportPdf({
                    dataset,
                    target,
                    taskType,
                    testSize: Number(testSize),
                    cvFolds: Number(cvFolds),
                    imbalanceStrategy,
                    results,
                  });
                  setNotice("Branded modeling PDF downloaded.");
                }}
              >
                <Download className="h-4 w-4" />
                Export Modeling Report
              </button>
              <button
                className="ds-button-primary"
                onClick={async () => {
                  const message = await shareText(
                    `DataStory modeling for ${dataset.filename}`,
                    [
                      `# DataStory Modeling Insights - ${dataset.filename}`,
                      "",
                      `- Target: ${target || "Not selected"}`,
                      `- Task: ${taskType ? capitalize(taskType) : "Not detected"}`,
                      `- Models trained: ${validResults.length}`,
                      `- Best model: ${best ? displayModel(best.model_name) : "Not available"}`,
                      `- Best score: ${best ? primaryScoreLabel(best, taskType) : "Not available"}`,
                      "",
                      "## Leaderboard",
                      ...[...validResults].sort(compareResults(taskType)).map((row, index) => `${index + 1}. ${displayModel(row.model_name)} - ${primaryScoreLabel(row, taskType)}`),
                      "",
                      "## Generated Chart",
                      markdownBarChart(
                        taskType === "regression" ? "Model RMSE Comparison" : "Model Weighted F1 Comparison",
                        [...validResults].sort(compareResults(taskType)).map((row) => displayModel(row.model_name)),
                        [...validResults].sort(compareResults(taskType)).map((row) => Number(taskType === "regression" ? row.rmse : row.f1 ?? row.primary_score ?? 0)),
                        taskType === "regression" ? "RMSE" : "Weighted F1",
                      ),
                    ].join("\n"),
                  );
                  setNotice(message);
                }}
              >
                <Share2 className="h-4 w-4" />
                Share Insights
              </button>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-[1.2fr_.8fr_.8fr_.8fr] gap-4">
            <label className="block">
              <span className="text-xs font-bold text-zinc-500">Target Column</span>
              <select
                className="mt-2 w-full rounded-lg border border-line bg-white px-4 py-3 font-bold dark:bg-zinc-900"
                value={target}
                onChange={(event) => {
                  setTarget(event.target.value);
                  setTaskType("");
                  setImbalanceInfo(null);
                  setImbalanceStrategy("none");
                  setSelectedModels([]);
                }}
              >
                <option value="">Choose a target</option>
                {dataset.column_names.map((column) => (
                  <option key={column} value={column}>{column}</option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="text-xs font-bold text-zinc-500">Test Size</span>
              <select className="mt-2 w-full rounded-lg border border-line bg-white px-4 py-3 dark:bg-zinc-900" value={testSize} onChange={(event) => setTestSize(event.target.value)}>
                <option value="0.2">20%</option>
                <option value="0.25">25%</option>
                <option value="0.3">30%</option>
              </select>
            </label>
            <label className="block">
              <span className="text-xs font-bold text-zinc-500">Scoring Metric</span>
              <select className="mt-2 w-full rounded-lg border border-line bg-white px-4 py-3 dark:bg-zinc-900" value={taskType === "regression" ? "RMSE" : "Weighted F1"} disabled>
                <option>{taskType === "regression" ? "RMSE" : "Weighted F1"}</option>
              </select>
            </label>
            <label className="block">
              <span className="text-xs font-bold text-zinc-500">Cross Validation</span>
              <select className="mt-2 w-full rounded-lg border border-line bg-white px-4 py-3 dark:bg-zinc-900" value={cvFolds} onChange={(event) => setCvFolds(event.target.value)}>
                <option value="3">3-Fold CV</option>
                <option value="5">5-Fold CV</option>
                <option value="10">10-Fold CV</option>
              </select>
            </label>
          </div>

          {taskType === "classification" && imbalanceInfo && (
            <div
              className={`mt-5 rounded-xl border p-4 ${
                Boolean(imbalanceInfo.imbalanced ?? imbalanceInfo.is_imbalanced)
                  ? "border-amber-300 bg-[#FFF7E8] text-[#3d2d13] dark:border-amber-700/70 dark:bg-[#2b2418] dark:text-amber-50"
                  : "border-[#cbdac3] bg-[#F3F8EF] text-[#263222] dark:border-[#506b45] dark:bg-[#1f2d1c] dark:text-[#e7f1df]"
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex gap-3">
                  <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-sage dark:text-[#a9ca91]" />
                  <div>
                    <h3 className="font-black">Class Imbalance Handling</h3>
                    <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-200">
                      {Boolean(imbalanceInfo.imbalanced ?? imbalanceInfo.is_imbalanced)
                        ? "DataStory detected class imbalance. You can apply SMOTE safely inside the training pipeline, after the train/test split."
                        : "No strong class imbalance was detected for this target."}
                    </p>
                    {imbalanceInfo.class_distribution && (
                      <div className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
                        Distribution: {Object.entries(imbalanceInfo.class_distribution).map(([key, value]) => `${key}: ${value}`).join(", ")}
                      </div>
                    )}
                  </div>
                </div>
                <select
                  className="rounded-lg border border-line bg-white px-4 py-2 text-sm font-bold dark:bg-zinc-900"
                  value={imbalanceStrategy}
                  onChange={(event) => setImbalanceStrategy(event.target.value as "none" | "smote" | "class_weight")}
                >
                  <option value="none">No resampling</option>
                  <option value="smote">Use SMOTE</option>
                  <option value="class_weight">Use class weights</option>
                </select>
              </div>
            </div>
          )}

          <div className="mt-6 rounded-xl border border-line bg-stone-50/70 p-4 dark:bg-zinc-900/60">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="font-black">Choose Models to Train</h3>
                <p className="mt-1 text-sm text-zinc-500">
                  {taskType
                    ? `${selectedModels.length} of ${availableModels.length} compatible ${taskType} models selected.`
                    : "Detect the task type first so DataStory can show compatible models."}
                </p>
              </div>
              <div className="flex gap-2">
                <button
                  className="ds-button-secondary"
                  disabled={!availableModels.length}
                  onClick={() => setSelectedModels(availableModels)}
                >
                  <SlidersHorizontal className="h-4 w-4" />
                  Select All
                </button>
                <button
                  className="ds-button-secondary"
                  disabled={!selectedModels.length}
                  onClick={() => setSelectedModels([])}
                >
                  Clear
                </button>
              </div>
            </div>

            <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
              {models.isLoading && (
                <div className="col-span-full rounded-lg border border-dashed border-line bg-white p-5 text-center text-sm text-zinc-500 dark:bg-zinc-950">
                  Loading compatible models...
                </div>
              )}
              {!models.isLoading && availableModels.length === 0 && (
                <div className="col-span-full rounded-lg border border-dashed border-line bg-white p-5 text-center text-sm text-zinc-500 dark:bg-zinc-950">
                  Pick a target column, then use Detect Target Task to unlock model choices.
                </div>
              )}
              {availableModels.map((model) => {
                const selected = selectedModels.includes(model);
                const meta = MODEL_META[model] ?? { family: "Model", short: model };
                return (
                  <button
                    key={model}
                    className={`rounded-lg border bg-white p-4 text-left transition dark:bg-zinc-950 ${selected ? "border-sage ring-2 ring-[#DCEAD1]" : "border-line hover:border-sage/60"}`}
                    onClick={() => setSelectedModels((current) => selected ? current.filter((item) => item !== model) : [...current, model])}
                    type="button"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="font-black">{meta.short}</div>
                        <div className="mt-1 text-xs text-zinc-500">{meta.family}</div>
                      </div>
                      <span className={`grid h-6 w-6 shrink-0 place-items-center rounded-full border ${selected ? "border-sage bg-sage text-white" : "border-line text-transparent"}`}>
                        <Check className="h-3.5 w-3.5" />
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="mt-5 flex gap-3">
            <button className="ds-button-secondary" disabled={!target || detect.isLoading} onClick={() => detect.mutate()}>
              <Target className="h-4 w-4" />
              Detect Target Task
            </button>
            <button className="ds-button-primary" disabled={!taskType || !selectedModels.length || train.isLoading} onClick={() => train.mutate()}>
              <RefreshCw className={`h-4 w-4 ${train.isLoading ? "animate-spin" : ""}`} />
              {train.isLoading ? "Training Models" : "Train Selected Models"}
            </button>
            {validResults.length > 0 && (
              <button className="ds-button-secondary" onClick={() => navigate("/results")}>
                View Results
                <ChevronRight className="h-4 w-4" />
              </button>
            )}
          </div>
          {detect.error instanceof Error && <div className="mt-3 text-sm font-bold text-red-600">{detect.error.message}</div>}
          {train.error instanceof Error && <div className="mt-3 text-sm font-bold text-red-600">{train.error.message}</div>}
        </section>

        <section className="grid grid-cols-5 gap-4">
          {metricCards.map(([label, value, caption, Icon]) => (
            <div key={label} className="ds-card p-5">
              <div className="flex items-center gap-4">
                <span className="grid h-11 w-11 place-items-center rounded-full bg-[#EEF5E9] text-sage"><Icon className="h-5 w-5" /></span>
                <div>
                  <div className="text-sm text-zinc-500">{label}</div>
                  <div className="mt-1 text-xl font-black">{value}</div>
                  <div className="text-xs text-zinc-500">{caption}</div>
                </div>
              </div>
            </div>
          ))}
        </section>

        <section className="grid grid-cols-[1.1fr_.9fr] gap-5">
          <div className="ds-card p-5">
            <h3 className="font-black">Model Performance Summary</h3>
            <p className="mt-1 text-sm text-zinc-500">Performance of trained models on the test set.</p>
            <div className="mt-4 overflow-hidden rounded-xl border border-line">
              <table className="w-full text-left text-sm">
                <thead className="bg-stone-50 text-zinc-500 dark:bg-zinc-800">
                  <tr>
                    <th className="p-3">Model</th>
                    {taskType === "regression" ? (
                      <>
                        <th className="p-3">RMSE</th>
                        <th className="p-3">MAE</th>
                        <th className="p-3">R2</th>
                        <th className="p-3">CV RMSE</th>
                      </>
                    ) : (
                      <>
                        <th className="p-3">Accuracy</th>
                        <th className="p-3">Precision</th>
                        <th className="p-3">Recall</th>
                        <th className="p-3">Weighted F1</th>
                        <th className="p-3">ROC-AUC</th>
                      </>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {(results.length ? results : selectedModels.map((model) => ({ model_name: model } as TrainResult))).map((row, index) => (
                    <tr key={row.model_name} className="border-t border-line">
                      <td className="p-3 font-bold">
                        <span className="mr-2 inline-grid h-5 w-5 place-items-center rounded-full bg-sage text-[11px] text-white">{index + 1}</span>
                        {displayModel(row.model_name)}
                        {best?.model_name === row.model_name && <span className="ml-2 rounded-full bg-[#EEF5E9] px-2 py-1 text-xs text-sage">Best</span>}
                        {row.error && <div className="mt-1 text-xs text-red-600">{row.error}</div>}
                      </td>
                      {taskType === "regression" ? (
                        <>
                          <td className="p-3">{formatMetric(row.rmse)}</td>
                          <td className="p-3">{formatMetric(row.mae)}</td>
                          <td className="p-3">{formatMetric(row.r2)}</td>
                          <td className="p-3">{formatMetric(row.cv_score)}</td>
                        </>
                      ) : (
                        <>
                          <td className="p-3">{formatMetric(row.accuracy)}</td>
                          <td className="p-3">{formatMetric(row.precision)}</td>
                          <td className="p-3">{formatMetric(row.recall)}</td>
                          <td className="p-3">{formatMetric(row.f1)}</td>
                          <td className="p-3">{formatMetric(row.roc_auc)}</td>
                        </>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="ds-card p-5">
            <div className="flex items-center gap-2">
              <h3 className="font-black">Best Model Details</h3>
              {best && <span className="rounded-full bg-[#EEF5E9] px-3 py-1 text-xs font-bold text-sage">{displayModel(best.model_name)}</span>}
            </div>
            {best ? (
              <>
                <div className="mt-5 grid grid-cols-3 gap-3 text-sm">
                  <Score label={taskType === "regression" ? "RMSE" : "Weighted F1"} value={taskType === "regression" ? best.rmse : best.f1} />
                  <Score label={taskType === "regression" ? "R2" : "Accuracy"} value={taskType === "regression" ? best.r2 : best.accuracy} />
                  <Score label={taskType === "regression" ? "MAE" : "ROC-AUC"} value={taskType === "regression" ? best.mae : best.roc_auc} />
                </div>
                <div className="mt-5 rounded-xl bg-stone-50 p-4 text-sm leading-6 dark:bg-zinc-900">
                  {best.reason ?? `${displayModel(best.model_name)} currently has the strongest primary score among the trained models.`}
                </div>
                <div className="mt-5">
                  <h4 className="text-sm font-black">Best Parameters</h4>
                  <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                    {Object.entries(best.best_params ?? {}).length ? Object.entries(best.best_params ?? {}).map(([key, value]) => (
                      <div key={key} className="rounded-lg border border-line p-3">
                        <div className="font-bold text-zinc-500">{key}</div>
                        <div className="mt-1 font-black">{String(value)}</div>
                      </div>
                    )) : <div className="rounded-lg border border-line p-3 text-zinc-500">No tuned parameters for this model.</div>}
                  </div>
                </div>
              </>
            ) : (
              <div className="mt-5 rounded-xl bg-stone-50 p-8 text-center text-sm text-zinc-500 dark:bg-zinc-900">
                Select a target, choose models, then train to see best model details.
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
function compareResults(taskType: string) {
  return (a: TrainResult, b: TrainResult) => {
    if (taskType === "regression") return (a.rmse ?? Number.POSITIVE_INFINITY) - (b.rmse ?? Number.POSITIVE_INFINITY);
    return (b.primary_score ?? 0) - (a.primary_score ?? 0);
  };
}

function displayModel(model: string) {
  return MODEL_META[model]?.short ?? model;
}

function formatMetric(value?: number | null) {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

function formatPrimaryScore(row: TrainResult, taskType: string) {
  if (taskType === "regression") return row.rmse ? row.rmse.toFixed(3) : "-";
  return row.f1 ? row.f1.toFixed(3) : row.primary_score ? row.primary_score.toFixed(3) : "-";
}

function primaryScoreLabel(row: TrainResult, taskType: string) {
  if (taskType === "regression") return `RMSE ${formatMetric(row.rmse)}`;
  return `Weighted F1 ${formatMetric(row.f1)}`;
}

function capitalize(value: string) {
  return value ? value[0].toUpperCase() + value.slice(1) : value;
}

function Score({ label, value }: { label: string; value?: number | null }) {
  return (
    <div>
      <div className="text-xs font-bold text-zinc-500">{label}</div>
      <div className="mt-1 text-lg font-black text-sage">{formatMetric(value)}</div>
    </div>
  );
}
