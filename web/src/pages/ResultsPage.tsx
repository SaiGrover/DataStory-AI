import { Link } from "react-router-dom";
import {
  AlertTriangle,
  ArrowRight,
  Award,
  BarChart3,
  Download,
  FileText,
  Medal,
  SlidersHorizontal,
  Target,
  Trophy,
} from "lucide-react";
import { useMemo } from "react";
import { downloadTextFile } from "../lib/browserActions";
import { useAppState } from "../lib/store";
import type { TrainResult } from "../lib/types";

export function ResultsPage() {
  const { dataset, target, taskType, results } = useAppState();
  const validResults = useMemo(() => results.filter((row) => !row.error), [results]);
  const failedResults = useMemo(() => results.filter((row) => row.error), [results]);
  const rankedResults = useMemo(() => [...validResults].sort(compareResults(taskType)), [taskType, validResults]);
  const best = rankedResults[0] ?? null;
  const maxRmse = Math.max(...rankedResults.map((row) => row.rmse ?? 0), 0);

  if (!results.length) {
    return (
      <section className="ds-card grid min-h-[420px] place-items-center p-10 text-center">
        <div className="max-w-xl">
          <span className="mx-auto grid h-16 w-16 place-items-center rounded-full bg-[#EEF5E9] text-sage">
            <Trophy className="h-8 w-8" />
          </span>
          <h1 className="mt-5 text-3xl font-black">No leaderboard yet</h1>
          <p className="mt-3 text-sm leading-6 text-zinc-600 dark:text-zinc-300">
            Train a few models first, then this page will rank them by the right metric for your task.
          </p>
          <Link className="ds-button-primary mt-6" to="/models">
            Go to Modeling
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </section>
    );
  }

  return (
    <div className="space-y-5">
      <section className="ds-card overflow-hidden">
        <div className="grid gap-6 bg-[#F5F8F0] p-6 dark:bg-zinc-900 lg:grid-cols-[1fr_340px]">
          <div>
            <div className="flex items-center gap-3">
              <span className="grid h-14 w-14 place-items-center rounded-full bg-sage text-white">
                <Trophy className="h-7 w-7" />
              </span>
              <div>
                <p className="text-xs font-black uppercase tracking-wide text-sage">Model Results</p>
                <h1 className="text-3xl font-black">Leaderboard</h1>
              </div>
            </div>
            <p className="mt-4 max-w-3xl text-sm leading-6 text-zinc-600 dark:text-zinc-300">
              Compare every trained model, inspect the winning run, and export the summary for your AI report.
            </p>
            <div className="mt-5 flex flex-wrap gap-3 text-xs font-bold">
              <span className="rounded-full border border-line bg-white px-3 py-1.5 dark:bg-zinc-950">{dataset?.filename ?? "Dataset loaded"}</span>
              <span className="rounded-full border border-line bg-white px-3 py-1.5 dark:bg-zinc-950">Target: {target || "Not selected"}</span>
              <span className="rounded-full border border-line bg-white px-3 py-1.5 dark:bg-zinc-950">{metricLabel(taskType)}</span>
            </div>
          </div>

          <div className="rounded-xl border border-line bg-white p-5 dark:bg-zinc-950">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xs font-black uppercase tracking-wide text-zinc-500">Best Model</p>
                <h2 className="mt-2 text-2xl font-black">{best ? displayModel(best.model_name) : "No successful run"}</h2>
              </div>
              <span className="grid h-11 w-11 place-items-center rounded-full bg-[#EEF5E9] text-sage">
                <Award className="h-6 w-6" />
              </span>
            </div>
            <div className="mt-5 grid grid-cols-2 gap-3">
              <MiniStat label={metricLabel(taskType)} value={best ? formatPrimary(best, taskType) : "-"} />
              <MiniStat label="Models Trained" value={validResults.length.toString()} />
            </div>
            <button className="ds-button-secondary mt-5 w-full" onClick={() => exportLeaderboard(dataset?.filename ?? "dataset", rankedResults)}>
              <Download className="h-4 w-4" />
              Export Results
            </button>
          </div>
        </div>
      </section>

      <div className="grid gap-4 md:grid-cols-4">
        <MetricTile icon={Medal} label="Winner" value={best ? displayModel(best.model_name) : "-"} caption={best ? formatPrimary(best, taskType) : "No successful result"} />
        <MetricTile icon={Target} label="Target" value={target || "-"} caption={taskType ? capitalize(taskType) : "Task not selected"} />
        <MetricTile icon={BarChart3} label="Successful Runs" value={validResults.length.toString()} caption={`${failedResults.length} failed`} />
        <MetricTile icon={SlidersHorizontal} label="Primary Metric" value={metricLabel(taskType)} caption={taskType === "regression" ? "Lower is better" : "Higher is better"} />
      </div>

      <section className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="ds-card overflow-hidden">
          <div className="flex items-center justify-between gap-4 border-b border-line px-5 py-4">
            <div>
              <h2 className="text-xl font-black">Ranked Models</h2>
              <p className="mt-1 text-sm text-zinc-500">Sorted by {metricLabel(taskType).toLowerCase()}.</p>
            </div>
            <Link className="ds-button-secondary" to="/report">
              <FileText className="h-4 w-4" />
              AI Report
            </Link>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[860px] text-left text-sm">
              <thead className="bg-stone-50 text-xs uppercase tracking-wide text-zinc-500 dark:bg-zinc-800">
                <tr>
                  <th className="px-5 py-4">Rank</th>
                  <th className="px-5 py-4">Model</th>
                  <th className="px-5 py-4">{metricLabel(taskType)}</th>
                  <th className="px-5 py-4">Score Bar</th>
                  <th className="px-5 py-4">{taskType === "regression" ? "MAE" : "Accuracy"}</th>
                  <th className="px-5 py-4">{taskType === "regression" ? "R2" : "Precision"}</th>
                  <th className="px-5 py-4">{taskType === "regression" ? "RMSE" : "Recall"}</th>
                </tr>
              </thead>
              <tbody>
                {rankedResults.map((row, index) => (
                  <tr key={row.model_name} className="border-t border-line align-middle">
                    <td className="px-5 py-4">
                      <span className={`grid h-8 w-8 place-items-center rounded-full text-xs font-black ${index === 0 ? "bg-sage text-white" : "bg-stone-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-200"}`}>
                        {index + 1}
                      </span>
                    </td>
                    <td className="px-5 py-4">
                      <div className="font-black">{displayModel(row.model_name)}</div>
                      {index === 0 && <span className="mt-1 inline-flex rounded-full bg-[#EEF5E9] px-2 py-0.5 text-[11px] font-bold text-sage">Best</span>}
                    </td>
                    <td className="px-5 py-4 font-black">{formatPrimary(row, taskType)}</td>
                    <td className="px-5 py-4">
                      <div className="h-2.5 w-full overflow-hidden rounded-full bg-stone-100 dark:bg-zinc-800">
                        <div className="h-full rounded-full bg-sage" style={{ width: `${scoreWidth(row, taskType, maxRmse)}%` }} />
                      </div>
                    </td>
                    <td className="px-5 py-4">{formatMetric(taskType === "regression" ? row.mae : row.accuracy)}</td>
                    <td className="px-5 py-4">{formatMetric(taskType === "regression" ? row.r2 : row.precision)}</td>
                    <td className="px-5 py-4">{formatMetric(taskType === "regression" ? row.rmse : row.recall)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {!rankedResults.length && (
            <div className="p-8 text-center text-sm text-zinc-500">
              No successful model runs were returned. Check the failed runs panel for the training errors.
            </div>
          )}
        </div>

        <aside className="space-y-5">
          <section className="ds-card p-5">
            <h2 className="text-lg font-black">Best Model Details</h2>
            {best ? (
              <>
                <div className="mt-4 rounded-xl border border-line bg-[#F8FAF5] p-4 dark:bg-zinc-950">
                  <div className="text-xs font-bold uppercase tracking-wide text-zinc-500">{displayModel(best.model_name)}</div>
                  <div className="mt-2 text-3xl font-black text-sage">{formatPrimary(best, taskType)}</div>
                  <p className="mt-3 text-sm leading-6 text-zinc-600 dark:text-zinc-300">{best.reason ?? defaultReason(best, taskType)}</p>
                </div>
                <h3 className="mt-5 text-sm font-black">Best Parameters</h3>
                <div className="mt-3 space-y-2">
                  {best.best_params && Object.keys(best.best_params).length ? (
                    Object.entries(best.best_params).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between gap-4 rounded-lg border border-line px-3 py-2 text-xs">
                        <span className="font-bold text-zinc-500">{key}</span>
                        <span className="text-right font-black">{String(value)}</span>
                      </div>
                    ))
                  ) : (
                    <div className="rounded-lg border border-line p-3 text-sm text-zinc-500">No tuned parameters for this model.</div>
                  )}
                </div>
              </>
            ) : (
              <p className="mt-3 text-sm text-zinc-500">No successful model run is available yet.</p>
            )}
          </section>

          {failedResults.length > 0 && (
            <section className="ds-card p-5">
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-[#C9822A]" />
                <h2 className="text-lg font-black">Failed Runs</h2>
              </div>
              <div className="mt-4 space-y-3">
                {failedResults.map((row) => (
                  <div key={row.model_name} className="rounded-xl border border-line p-3">
                    <div className="font-black">{displayModel(row.model_name)}</div>
                    <p className="mt-1 text-xs leading-5 text-zinc-500">{row.error}</p>
                  </div>
                ))}
              </div>
            </section>
          )}
        </aside>
      </section>
    </div>
  );
}

function compareResults(taskType: string) {
  return (a: TrainResult, b: TrainResult) => {
    if (taskType === "regression") return (a.rmse ?? Number.POSITIVE_INFINITY) - (b.rmse ?? Number.POSITIVE_INFINITY);
    return (b.f1 ?? b.primary_score ?? 0) - (a.f1 ?? a.primary_score ?? 0);
  };
}

function metricLabel(taskType: string) {
  return taskType === "regression" ? "RMSE" : "Weighted F1";
}

function formatPrimary(row: TrainResult, taskType: string) {
  return formatMetric(taskType === "regression" ? row.rmse : row.f1 ?? row.primary_score);
}

function scoreWidth(row: TrainResult, taskType: string, maxRmse: number) {
  if (taskType === "regression") {
    if (!row.rmse || !maxRmse) return 8;
    return Math.max(8, Math.min(100, 100 - (row.rmse / maxRmse) * 72));
  }
  return Math.max(8, Math.min(100, (row.f1 ?? row.primary_score ?? 0) * 100));
}

function formatMetric(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "-";
}

function displayModel(model: string) {
  return model.replace(/_/g, " ");
}

function capitalize(value: string) {
  return value ? value.charAt(0).toUpperCase() + value.slice(1) : "";
}

function defaultReason(row: TrainResult, taskType: string) {
  if (taskType === "regression") return `${displayModel(row.model_name)} has the lowest RMSE among the successful trained models.`;
  return `${displayModel(row.model_name)} has the highest weighted F1 among the successful trained models.`;
}

function exportLeaderboard(filename: string, results: TrainResult[]) {
  const lines = [
    `DataStory AI Leaderboard - ${filename}`,
    `Generated: ${new Date().toLocaleString()}`,
    "",
    ...results.map((row, index) => `${index + 1}. ${row.model_name}: primary=${row.primary_score ?? "-"} f1=${row.f1 ?? "-"} accuracy=${row.accuracy ?? "-"} rmse=${row.rmse ?? "-"}`),
  ];
  downloadTextFile(`${filename.replace(/\W+/g, "_")}_leaderboard.txt`, lines.join("\n"));
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-line p-3">
      <div className="text-[11px] font-bold uppercase tracking-wide text-zinc-500">{label}</div>
      <div className="mt-1 text-lg font-black">{value}</div>
    </div>
  );
}

function MetricTile({ icon: Icon, label, value, caption }: { icon: typeof Trophy; label: string; value: string; caption: string }) {
  return (
    <div className="ds-card p-5">
      <span className="grid h-11 w-11 place-items-center rounded-full bg-[#EEF5E9] text-sage">
        <Icon className="h-5 w-5" />
      </span>
      <div className="mt-4 text-sm text-zinc-500">{label}</div>
      <div className="mt-1 truncate text-2xl font-black">{value}</div>
      <div className="mt-2 text-xs text-zinc-500">{caption}</div>
    </div>
  );
}
