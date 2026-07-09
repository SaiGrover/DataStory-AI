import type { DatasetResponse, SampleDataset, TrainResult } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function listSamples() {
  return request<{ samples: SampleDataset[] }>("/samples");
}

export function loadSample(sampleId: string) {
  return request<DatasetResponse>(`/samples/${sampleId}`, { method: "POST" });
}

export function uploadCsv(file: File) {
  const form = new FormData();
  form.append("file", file);
  return request<DatasetResponse>("/upload", {
    method: "POST",
    body: form,
  });
}

export type CleaningConfig = {
  remove_dups?: boolean;
  drop_empty_rows?: boolean;
  drop_high_missing?: boolean;
  remove_constant?: boolean;
  numeric_fill?: "None" | "Mean" | "Median" | "Custom value";
  cat_fill?: "None" | "Mode" | "Custom value";
  strip_whitespace?: boolean;
  text_case?: "Keep as-is" | "Lowercase" | "Uppercase" | "Title Case";
  coerce_numeric_text?: boolean;
  convert_dates?: boolean;
  outlier_action?: "None" | "Cap at IQR bounds (1.5x)" | "Remove outlier rows";
  drop_duplicate_columns?: boolean;
  numeric_columns?: string[];
};

export type CleaningResult = {
  rows: number;
  columns: number;
  missing: number;
  duplicates: number;
  actions?: string[];
};

function cleanRequestBody(previewOnly: boolean, config?: CleaningConfig) {
  return JSON.stringify({
    preview_only: previewOnly,
    config: config ?? {
      remove_dups: true,
      numeric_fill: "Median",
      cat_fill: "Mode",
      remove_constant: true,
    },
  });
}

export function previewCleaning(datasetId: string, config?: CleaningConfig) {
  return request<CleaningResult & { actions: string[] }>(`/clean/${datasetId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: cleanRequestBody(true, config),
  });
}

export function applyCleaning(datasetId: string, config?: CleaningConfig) {
  return request<DatasetResponse & CleaningResult>(`/clean/${datasetId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: cleanRequestBody(false, config),
  });
}

export type EdaResponse = {
  numeric_columns: string[];
  categorical_columns: string[];
  id_columns?: string[];
  correlation?: { columns: string[]; values: number[][] };
  numeric_profiles: Array<{
    column: string;
    histogram: { bins: number[]; counts: number[] };
    box: { min: number; q1: number; median: number; q3: number; max: number; outliers: number };
  }>;
  categorical_profiles: Array<{ column: string; labels: string[]; counts: number[] }>;
};

export function getEda(datasetId: string) {
  return request<EdaResponse>(`/eda/${datasetId}`);
}

export function selectTarget(datasetId: string, target: string) {
  return request<{ target: string; task_type: "classification" | "regression"; imbalance_info?: unknown }>(
    `/select-target/${datasetId}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target }),
    },
  );
}

export function listModels(taskType: string) {
  return request<{ task_type: string; models: string[] }>(`/models?task_type=${taskType}`);
}

export function trainModels(
  datasetId: string,
  target: string,
  taskType: string,
  modelNames: string[],
  options?: { testSize?: number; cvFolds?: number },
) {
  return request<{ results: TrainResult[] }>(`/train/${datasetId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      target,
      task_type: taskType,
      model_names: modelNames,
      imbalance_strategy: null,
      cv_folds: options?.cvFolds ?? 3,
      test_size: options?.testSize ?? 0.2,
    }),
  });
}
