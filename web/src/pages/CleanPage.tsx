import { useMemo, useState } from "react";
import { useMutation } from "react-query";
import { useNavigate } from "react-router-dom";
import type { LucideIcon } from "lucide-react";
import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Bot,
  CalendarDays,
  Check,
  ChevronDown,
  Columns3,
  Database,
  Eraser,
  Eye,
  FileText,
  Grid3X3,
  ListChecks,
  RefreshCw,
  Rows3,
  Scissors,
  Settings,
  SlidersHorizontal,
  Sparkles,
  Table2,
  Wand2,
} from "lucide-react";
import { applyCleaning, previewCleaning, type CleaningConfig } from "../lib/api";
import { useAppState } from "../lib/store";

type CleaningOption = {
  id: string;
  label: string;
  implemented?: boolean;
  defaultSelected?: boolean;
  config?: Partial<CleaningConfig>;
};

type CleaningGroup = {
  id: string;
  title: string;
  icon: LucideIcon;
  tint: string;
  options: CleaningOption[];
};

const groups: CleaningGroup[] = [
  {
    id: "missing",
    title: "Missing Values",
    icon: AlertTriangle,
    tint: "bg-[#EEF5E9]",
    options: [
      { id: "num_mean", label: "Fill numeric with Mean", implemented: true, defaultSelected: true, config: { numeric_fill: "Mean" } },
      { id: "num_median", label: "Fill numeric with Median", implemented: true, defaultSelected: true, config: { numeric_fill: "Median" } },
      { id: "num_mode", label: "Fill numeric with Mode", defaultSelected: true },
      { id: "knn", label: "Fill using KNN Imputer" },
      { id: "mice", label: "Fill using Iterative Imputer (MICE)" },
      { id: "forward", label: "Forward Fill (time series)" },
      { id: "backward", label: "Backward Fill (time series)" },
      { id: "custom_fill", label: "Fill with custom value", implemented: true, config: { numeric_fill: "Custom value", cat_fill: "Custom value" } },
      { id: "drop_missing_rows", label: "Drop rows containing missing values", implemented: true, config: { drop_empty_rows: true } },
      { id: "drop_high_missing", label: "Drop columns exceeding X% missing values", implemented: true, defaultSelected: true, config: { drop_high_missing: true } },
    ],
  },
  {
    id: "duplicates",
    title: "Duplicate Handling",
    icon: Rows3,
    tint: "bg-[#FBF3DF]",
    options: [
      { id: "remove_dups", label: "Remove exact duplicate rows", implemented: true, defaultSelected: true, config: { remove_dups: true } },
      { id: "dups_selected_cols", label: "Remove duplicates based on selected columns" },
      { id: "keep_first", label: "Keep first occurrence" },
      { id: "keep_last", label: "Keep last occurrence" },
      { id: "preview_dups", label: "Show duplicate preview before deletion" },
    ],
  },
  {
    id: "types",
    title: "Data Type Fixes",
    icon: Table2,
    tint: "bg-[#ECF4F8]",
    options: [
      { id: "detect_types", label: "Detect incorrect data types", defaultSelected: true },
      { id: "to_numeric", label: "Convert to Numeric", implemented: true, config: { coerce_numeric_text: true } },
      { id: "to_category", label: "Convert to Category" },
      { id: "to_boolean", label: "Convert to Boolean" },
      { id: "to_datetime", label: "Convert to Datetime", implemented: true, config: { convert_dates: true } },
      { id: "mixed_types", label: "Detect mixed-type columns" },
      { id: "infer_types", label: "Infer better data types automatically" },
    ],
  },
  {
    id: "text",
    title: "Text Cleaning",
    icon: Eraser,
    tint: "bg-[#F4EDF8]",
    options: [
      { id: "strip_spaces", label: "Strip leading/trailing spaces", implemented: true, defaultSelected: true, config: { strip_whitespace: true } },
      { id: "lowercase", label: "Convert to lowercase", implemented: true, config: { text_case: "Lowercase" } },
      { id: "uppercase", label: "Convert to uppercase", implemented: true, config: { text_case: "Uppercase" } },
      { id: "title_case", label: "Title Case", implemented: true, config: { text_case: "Title Case" } },
      { id: "extra_spaces", label: "Remove extra spaces" },
      { id: "punctuation", label: "Remove punctuation" },
      { id: "special_chars", label: "Remove special characters" },
      { id: "html_tags", label: "Remove HTML tags" },
      { id: "emojis", label: "Remove emojis" },
      { id: "unicode", label: "Normalize unicode characters" },
    ],
  },
  {
    id: "categorical",
    title: "Categorical Cleaning",
    icon: Grid3X3,
    tint: "bg-[#EEF5E9]",
    options: [
      { id: "standardize_categories", label: "Standardize categories", defaultSelected: true },
      { id: "merge_similar", label: "Merge similar categories" },
      { id: "male_variants", label: "Male / male / M -> Male" },
      { id: "spelling", label: "Fix common spelling mistakes" },
      { id: "rare_other", label: 'Replace rare categories with "Other"' },
      { id: "yes_no", label: "Encode Yes/No variations" },
      { id: "trim_category", label: "Trim whitespace", implemented: true, config: { strip_whitespace: true } },
    ],
  },
  {
    id: "outliers",
    title: "Outlier Handling",
    icon: BarChart3,
    tint: "bg-[#FBEDE5]",
    options: [
      { id: "detect_outliers", label: "Detect Outliers", defaultSelected: true },
      { id: "remove_iqr", label: "Remove using IQR", implemented: true, config: { outlier_action: "Remove outlier rows" } },
      { id: "remove_zscore", label: "Remove using Z-score" },
      { id: "winsor", label: "Winsorization (cap extreme values)", implemented: true, defaultSelected: true, config: { outlier_action: "Cap at IQR bounds (1.5x)" } },
      { id: "clip_percentile", label: "Clip values to percentile" },
      { id: "show_affected", label: "Show affected rows before applying" },
    ],
  },
  {
    id: "scaling",
    title: "Feature Scaling",
    icon: SlidersHorizontal,
    tint: "bg-[#ECF4F8]",
    options: [
      { id: "standard_scaler", label: "StandardScaler" },
      { id: "minmax", label: "Min-Max Scaling" },
      { id: "robust", label: "RobustScaler" },
      { id: "maxabs", label: "MaxAbsScaler" },
      { id: "normalize", label: "Normalize vectors (L1/L2)" },
    ],
  },
  {
    id: "encoding",
    title: "Encoding",
    icon: ListChecks,
    tint: "bg-[#F4EDF8]",
    options: [
      { id: "onehot", label: "One-Hot Encoding" },
      { id: "label", label: "Label Encoding" },
      { id: "ordinal", label: "Ordinal Encoding" },
      { id: "frequency", label: "Frequency Encoding" },
      { id: "target_encoding", label: "Target Encoding (warning about leakage)" },
      { id: "binary", label: "Binary Encoding" },
    ],
  },
  {
    id: "datetime",
    title: "Date & Time",
    icon: CalendarDays,
    tint: "bg-[#FBF3DF]",
    options: [
      { id: "detect_dates", label: "Detect date columns", implemented: true, defaultSelected: true, config: { convert_dates: true } },
      { id: "parse_dates", label: "Parse date formats", implemented: true, config: { convert_dates: true } },
      { id: "year", label: "Extract Year" },
      { id: "month", label: "Extract Month" },
      { id: "day", label: "Extract Day" },
      { id: "weekday", label: "Extract Weekday" },
      { id: "quarter", label: "Extract Quarter" },
      { id: "age_from_date", label: "Calculate Age from Date" },
      { id: "time_diff", label: "Calculate Time Difference" },
    ],
  },
  {
    id: "columns",
    title: "Column Operations",
    icon: Columns3,
    tint: "bg-[#EEF5E9]",
    options: [
      { id: "rename", label: "Rename columns" },
      { id: "remove_constant", label: "Remove constant columns", implemented: true, config: { remove_constant: true } },
      { id: "low_variance", label: "Remove low variance columns" },
      { id: "id_columns", label: "Remove ID columns" },
      { id: "drop_selected", label: "Drop selected columns" },
      { id: "reorder", label: "Reorder columns" },
      { id: "merge_columns", label: "Merge multiple columns" },
      { id: "split_column", label: "Split column by delimiter" },
    ],
  },
  {
    id: "numeric",
    title: "Numeric Cleaning",
    icon: Scissors,
    tint: "bg-[#FBEDE5]",
    options: [
      { id: "negative_invalid", label: "Remove negative values where invalid" },
      { id: "infinite", label: "Replace infinite values" },
      { id: "round", label: "Round decimals" },
      { id: "clip_range", label: "Clip values to range" },
      { id: "impossible", label: "Detect impossible values" },
      { id: "zero_variance", label: "Remove zero variance features", implemented: true, config: { remove_constant: true } },
    ],
  },
  {
    id: "correlation",
    title: "Correlation & Redundancy",
    icon: Sparkles,
    tint: "bg-[#ECF4F8]",
    options: [
      { id: "high_corr", label: "Remove highly correlated features" },
      { id: "vif", label: "Detect multicollinearity (VIF)" },
      { id: "duplicate_columns", label: "Remove duplicate columns", implemented: true, config: { drop_duplicate_columns: true } },
      { id: "leakage", label: "Detect leakage columns" },
    ],
  },
];

const defaultSelected = new Set(groups.flatMap((group) => group.options.filter((option) => option.defaultSelected).map((option) => option.id)));

type ImpactMetric = [string, number, number, LucideIcon];

export function CleanPage() {
  const { dataset, setDataset, setResults, setTarget, setTaskType } = useAppState();
  const navigate = useNavigate();
  const [selected, setSelected] = useState<Set<string>>(defaultSelected);
  const [activeGroup, setActiveGroup] = useState("all");
  const [showExpanded, setShowExpanded] = useState(false);
  const numericColumns = useMemo(
    () => (dataset?.profile.column_details ?? []).filter((col) => /int|float|double|number/i.test(String(col.Type))).map((col) => String(col.Column)),
    [dataset],
  );
  const [numericScope, setNumericScope] = useState<"all" | "selected">("all");
  const [selectedNumericColumns, setSelectedNumericColumns] = useState<string[]>([]);

  const selectedOptions = useMemo(() => groups.flatMap((group) => group.options).filter((option) => selected.has(option.id)), [selected]);
  const selectedCount = selected.size;
  const implementedSelected = selectedOptions.filter((option) => option.implemented).length;
  const totalOptions = groups.reduce((sum, group) => sum + group.options.length, 0);
  const visibleGroups = activeGroup === "all" ? groups : groups.filter((group) => group.id === activeGroup);

  const config = useMemo<CleaningConfig>(() => {
    const merged: CleaningConfig = {
      numeric_fill: "None",
      cat_fill: "None",
      text_case: "Keep as-is",
      outlier_action: "None",
    };
    selectedOptions.forEach((option) => Object.assign(merged, option.config ?? {}));
    if (selected.has("num_mean") && !selected.has("num_median")) merged.numeric_fill = "Mean";
    if (selected.has("num_median")) merged.numeric_fill = "Median";
    if (selected.has("custom_fill")) {
      merged.numeric_fill = "Custom value";
      merged.cat_fill = "Custom value";
    }
    if (selected.has("num_mode") || selected.has("standardize_categories")) merged.cat_fill = "Mode";
    if (numericScope === "selected" && selectedNumericColumns.length > 0) merged.numeric_columns = selectedNumericColumns;
    return merged;
  }, [numericScope, selected, selectedNumericColumns, selectedOptions]);

  const preview = useMutation(() => previewCleaning(dataset!.dataset_id, config));
  const apply = useMutation(() => applyCleaning(dataset!.dataset_id, config), {
    onSuccess: (cleaned) => {
      setDataset(cleaned);
      setResults([]);
      setTarget("");
      setTaskType("");
      preview.reset();
    },
  });

  if (!dataset) return <div className="ds-card p-8">No dataset loaded. Start with an upload or sample dataset.</div>;

  const toggle = (id: string) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else {
        if (["num_mean", "num_median", "num_mode", "custom_fill"].includes(id)) {
          ["num_mean", "num_median", "num_mode", "custom_fill"].forEach((fillId) => next.delete(fillId));
        }
        next.add(id);
      }
      return next;
    });
  };

  const autoSelectRecommended = () => {
    setSelected(new Set(groups.flatMap((group) => group.options.filter((option) => option.defaultSelected || option.implemented).map((option) => option.id))));
  };

  const resetAll = () => {
    setSelected(new Set());
    preview.reset();
  };

  const estimatedRows = preview.data?.rows ?? dataset.rows;
  const estimatedColumns = preview.data?.columns ?? dataset.columns;
  const estimatedMissing = preview.data?.missing ?? (dataset.profile.total_missing ?? 0);
  const estimatedDuplicates = preview.data?.duplicates ?? (dataset.profile.duplicates ?? 0);
  const numericFillSelected = selectedOptions.some((option) => ["num_mean", "num_median", "num_mode", "custom_fill"].includes(option.id));
  const numericSelectionInvalid = numericFillSelected && numericScope === "selected" && selectedNumericColumns.length === 0;

  return (
    <div className="grid grid-cols-[320px_minmax(0,1fr)] items-start gap-5">
      <aside className="ds-card sticky top-36 max-h-[calc(100vh-270px)] overflow-y-auto p-5 pb-8">
        <h2 className="text-xl font-black">Cleaning Actions</h2>
        <p className="mt-1 text-xs text-zinc-500">Select and configure the cleaning steps.</p>
        <div className="mt-5 space-y-2">
          <button
            className={`flex w-full items-center justify-between rounded-xl px-4 py-3 text-left text-sm font-bold ${activeGroup === "all" ? "bg-[#EEF5E9] text-sage" : "hover:bg-stone-50 dark:hover:bg-zinc-800"}`}
                onClick={() => {
                  setActiveGroup("all");
                  setShowExpanded(false);
                }}
          >
            <span className="flex items-center gap-3">
              <Grid3X3 className="h-4 w-4" />
              All Cleaning Options
            </span>
            <span className="rounded-full bg-white px-2 py-1 text-[10px] text-sage dark:bg-zinc-900">{selectedCount} selected</span>
          </button>
          {groups.map((group) => {
            const Icon = group.icon;
            const count = group.options.filter((option) => selected.has(option.id)).length;
            return (
              <button
                key={group.id}
                className={`flex w-full items-center justify-between rounded-xl px-4 py-3 text-left text-xs font-bold ${activeGroup === group.id ? "bg-[#EEF5E9] text-sage" : "hover:bg-stone-50 dark:hover:bg-zinc-800"}`}
                onClick={() => {
                  setActiveGroup(group.id);
                  setShowExpanded(true);
                }}
              >
                <span className="flex min-w-0 items-center gap-3">
                  <Icon className="h-4 w-4 shrink-0" />
                  <span className="truncate">{group.title}</span>
                </span>
                <span className="flex shrink-0 items-center gap-2 text-[10px] text-zinc-500">
                  {count} selected
                  <ArrowRight className="h-3 w-3" />
                </span>
              </button>
            );
          })}
        </div>
        <button className="ds-button-primary mt-6 w-full disabled:cursor-not-allowed disabled:opacity-60" onClick={() => preview.mutate()} disabled={preview.isLoading || numericSelectionInvalid}>
          <Eye className="h-4 w-4" />
          Preview Impact
        </button>
        <button className="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-lg bg-ink px-4 py-2.5 text-sm font-bold text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-60" onClick={() => apply.mutate()} disabled={apply.isLoading || numericSelectionInvalid}>
          <Wand2 className="h-4 w-4" />
          {apply.isLoading ? "Applying..." : "Confirm & Apply"}
        </button>
        <button className="ds-button-secondary mt-3 w-full" onClick={resetAll}>
          <RefreshCw className="h-4 w-4" />
          Reset All
        </button>
      </aside>

      <main className="space-y-5">
        <section className="ds-card p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h1 className="text-2xl font-black">All Cleaning Options</h1>
              <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-300">Choose and configure the cleaning actions you want to apply to your dataset.</p>
            </div>
            <div className="flex gap-3">
              <button className="ds-button-secondary" onClick={autoSelectRecommended}>
                <Sparkles className="h-4 w-4" />
                Auto Select Recommended
              </button>
              <button
                className="ds-button-secondary"
                onClick={() => {
                  setActiveGroup("all");
                  setShowExpanded(true);
                }}
              >
                <ChevronDown className="h-4 w-4" />
                Expand All
              </button>
            </div>
          </div>

          <div className="mt-5 grid grid-cols-3 gap-4">
            {visibleGroups.map((group, index) => {
              const Icon = group.icon;
              const groupSelected = group.options.filter((option) => selected.has(option.id)).length;
              const displayedOptions = showExpanded || activeGroup !== "all" ? group.options : group.options.slice(0, 5);
              return (
                <article key={group.id} className="rounded-xl border border-line bg-white p-4 shadow-[0_10px_28px_rgba(39,43,34,.04)] dark:bg-zinc-900">
                  <div className="mb-4 flex items-center justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <span className={`grid h-9 w-9 place-items-center rounded-full ${group.tint}`}>
                        <Icon className="h-5 w-5 text-sage" />
                      </span>
                      <h3 className="text-sm font-black">
                        {index + 1}. {group.title}
                      </h3>
                    </div>
                    <span className="rounded-full bg-[#EEF5E9] px-2.5 py-1 text-[10px] font-bold text-sage">{groupSelected} selected</span>
                  </div>
                  <div className="space-y-2">
                    {displayedOptions.map((option) => (
                      <label key={option.id} className="flex cursor-pointer items-start gap-2 text-xs font-semibold text-zinc-700 dark:text-zinc-200">
                        <span
                          className={`mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded border ${
                            selected.has(option.id) ? "border-sage bg-sage text-white" : "border-line bg-white dark:bg-zinc-950"
                          }`}
                        >
                          {selected.has(option.id) && <Check className="h-3 w-3" />}
                        </span>
                        <input type="checkbox" className="hidden" checked={selected.has(option.id)} onChange={() => toggle(option.id)} />
                        <span>{option.label}</span>
                      </label>
                    ))}
                  </div>
                  {group.options.length > displayedOptions.length && <div className="mt-3 text-xs font-bold text-zinc-500">+{group.options.length - displayedOptions.length} more options</div>}
                  <button
                    className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg border border-line py-2 text-xs font-bold hover:bg-stone-50 dark:hover:bg-zinc-800"
                    onClick={() => {
                      setActiveGroup(group.id);
                      setShowExpanded(true);
                    }}
                  >
                    <Settings className="h-3.5 w-3.5" />
                    Configure
                  </button>
                </article>
              );
            })}
          </div>

          {selectedOptions.some((option) => ["num_mean", "num_median", "num_mode", "custom_fill"].includes(option.id)) && (
            <div className="mt-5 rounded-xl border border-line bg-[#FBF7ED] p-4 dark:bg-zinc-900">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h3 className="text-sm font-black">Numeric Fill Columns</h3>
                  <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-300">Choose which numeric columns receive the selected fill strategy.</p>
                </div>
                <div className="flex rounded-lg border border-line bg-white p-1 text-xs font-bold dark:bg-zinc-950">
                  <button className={`rounded-md px-3 py-1.5 ${numericScope === "all" ? "bg-sage text-white" : "text-zinc-500"}`} onClick={() => setNumericScope("all")}>
                    All numeric
                  </button>
                  <button className={`rounded-md px-3 py-1.5 ${numericScope === "selected" ? "bg-sage text-white" : "text-zinc-500"}`} onClick={() => setNumericScope("selected")}>
                    Selected
                  </button>
                </div>
              </div>
              {numericScope === "selected" && (
                <div className="mt-4 grid grid-cols-4 gap-2">
                  {numericColumns.map((column) => (
                    <label key={column} className="flex cursor-pointer items-center gap-2 rounded-lg border border-line bg-white px-3 py-2 text-xs font-semibold dark:bg-zinc-950">
                      <input
                        type="checkbox"
                        checked={selectedNumericColumns.includes(column)}
                        onChange={() =>
                          setSelectedNumericColumns((current) => (current.includes(column) ? current.filter((item) => item !== column) : [...current, column]))
                        }
                      />
                      <span className="truncate">{column}</span>
                    </label>
                  ))}
                </div>
              )}
              {numericSelectionInvalid && <div className="mt-3 text-xs font-bold text-amber-700">Select at least one numeric column, or switch back to All numeric.</div>}
            </div>
          )}
        </section>

        <section className="grid grid-cols-[1fr_.75fr] gap-5">
          <div className="ds-card p-5">
            <h3 className="text-sm font-black">Estimated Impact After Cleaning</h3>
            <div className="mt-4 grid grid-cols-4 gap-3">
              {([
                ["Rows", dataset.rows, estimatedRows, Rows3],
                ["Columns", dataset.columns, estimatedColumns, Columns3],
                ["Missing Values", dataset.profile.total_missing ?? 0, estimatedMissing, AlertTriangle],
                ["Duplicate Rows", dataset.profile.duplicates ?? 0, estimatedDuplicates, FileText],
              ] satisfies ImpactMetric[]).map(([label, before, after, Icon]) => (
                <div key={String(label)} className="rounded-xl border border-line bg-stone-50 p-4 dark:bg-zinc-800">
                  <Icon className="mb-2 h-6 w-6 text-sage" />
                  <div className="text-xs font-bold">{String(label)}</div>
                  <div className="mt-2 flex items-center gap-2 text-sm">
                    <strong>{Number(before).toLocaleString()}</strong>
                    <ArrowRight className="h-4 w-4 text-zinc-400" />
                    <strong>{Number(after).toLocaleString()}</strong>
                  </div>
                  <div className={`mt-1 text-[11px] font-bold ${Number(before) - Number(after) > 0 ? "text-sage" : "text-zinc-500"}`}>
                    {Number(before) - Number(after) > 0 ? `-${(Number(before) - Number(after)).toLocaleString()}` : "No change"}
                  </div>
                </div>
              ))}
            </div>
            {preview.data && (
              <div className="mt-4 rounded-xl bg-[#EEF5E9] p-4">
                <h4 className="text-sm font-black text-sage">Previewed actions</h4>
                <ul className="mt-2 grid grid-cols-2 gap-x-5 gap-y-1 text-xs text-zinc-700">
                  {preview.data.actions.map((action) => (
                    <li key={action}>- {action}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <aside className="ds-card grid grid-cols-[1fr_95px] items-center gap-4 bg-[#F7FAFF] p-5 dark:bg-zinc-900">
            <div>
              <h3 className="text-lg font-black text-sage">Preview before applying</h3>
              <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-300">
                See how each cleaning step will affect your data. You can review changes and apply only what you want.
              </p>
              <div className="mt-4 text-xs font-bold text-zinc-500">
                {implementedSelected} backend-ready actions selected out of {selectedCount}. Full catalog: {totalOptions} options.
              </div>
              <button className="ds-button-primary mt-5" onClick={() => navigate("/explore")}>
                <Bot className="h-4 w-4" />
                Continue to Analysis
              </button>
              <button className="ml-3 mt-5 inline-flex items-center justify-center gap-2 rounded-lg bg-ink px-4 py-2.5 text-sm font-bold text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-60" onClick={() => apply.mutate()} disabled={apply.isLoading || numericSelectionInvalid}>
                <Wand2 className="h-4 w-4" />
                Apply Options
              </button>
              {apply.isSuccess && <div className="mt-3 text-xs font-bold text-sage">Cleaning applied to the active dataset.</div>}
              {apply.error instanceof Error && <div className="mt-3 text-xs font-bold text-red-600">{apply.error.message}</div>}
            </div>
            <img src="/pot_plant_small.png" alt="Potted plant" className="h-24 w-24 object-contain" />
          </aside>
        </section>
      </main>
    </div>
  );
}
