export type Profile = {
  rows?: number;
  columns?: number;
  numeric_cols?: number;
  categorical_cols?: number;
  date_cols?: number;
  bool_cols?: number;
  total_missing?: number;
  duplicates?: number;
  constant_cols?: string[];
  high_cardinality_cols?: string[];
  id_cols?: string[];
  column_details?: Array<Record<string, unknown>>;
  possible_targets?: string[];
  warnings?: Array<{ title: string; message: string }>;
  health_score?: number;
};

export type DatasetResponse = {
  dataset_id: string;
  filename: string;
  rows: number;
  columns: number;
  column_names: string[];
  profile: Profile;
};

export type SampleDataset = {
  id: string;
  name: string;
  description: string;
  rows: number;
  columns: number;
};

export type TrainResult = {
  model_name: string;
  primary_score?: number;
  cv_score?: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  roc_auc?: number | null;
  mae?: number;
  rmse?: number;
  r2?: number;
  error?: string;
  reason?: string;
  best_params?: Record<string, unknown>;
};
