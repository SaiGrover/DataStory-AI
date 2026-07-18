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
  categorical_columns?: string[];
  text_columns?: string[];
  outlier_columns?: string[];
  drop_columns?: string[];
  duplicate_key_columns?: string[];
  date_columns?: string[];
  type_columns?: string[];
  drop_missing_threshold?: number;
  custom_num_val?: number | string;
  custom_cat_val?: number | string;
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
    values?: number[];
    kde?: { x: number[]; y: number[] };
    box: { min: number; q1: number; median: number; q3: number; max: number; outliers: number };
    stats?: { mean: number; std: number; skew: number; missing: number };
  }>;
  categorical_profiles: Array<{ column: string; labels: string[]; counts: number[] }>;
};

export function getEda(datasetId: string) {
  return request<EdaResponse>(`/eda/${datasetId}`);
}

export type ImbalanceInfo = {
  imbalanced?: boolean;
  is_imbalanced?: boolean;
  class_distribution?: Record<string, number>;
  class_percentages?: Record<string, number>;
  minority_class?: string;
  majority_class?: string;
  min_percentage?: number;
  message?: string;
};

export function selectTarget(datasetId: string, target: string) {
  return request<{ target: string; task_type: "classification" | "regression"; imbalance_info?: ImbalanceInfo }>(
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
  options?: { testSize?: number; cvFolds?: number; imbalanceStrategy?: "none" | "smote" | "class_weight" | null },
) {
  return request<{ results: TrainResult[] }>(`/train/${datasetId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      target,
      task_type: taskType,
      model_names: modelNames,
      imbalance_strategy: options?.imbalanceStrategy && options.imbalanceStrategy !== "none" ? options.imbalanceStrategy : null,
      cv_folds: options?.cvFolds ?? 3,
      test_size: options?.testSize ?? 0.2,
    }),
  });
}

export function askDatasetAgent(
  datasetId: string,
  question: string,
  options?: {
    target?: string;
    taskType?: string;
    cleaningActions?: string[];
    history?: Array<{ role: "user" | "assistant"; content: string }>;
  },
) {
  return request<{
    answer: string;
    sources: string[];
    confidence: "high" | "medium" | "low";
    follow_up_questions: string[];
    mode: "ai-grounded" | "local-grounded";
  }>(`/chat/${datasetId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      target: options?.target,
      task_type: options?.taskType,
      cleaning_actions: options?.cleaningActions ?? [],
      history: options?.history ?? [],
    }),
  });
}
