# DataStory AI

**Your junior data analyst.** A Streamlit app that takes a dataset from upload to report:

**Upload → Data Health → Clean → Explore → Insights → Model → Results & Explainability → Ask the Assistant → Export a Report**

Every number DataStory AI shows (in the dashboard, the assistant's answers, and the exported reports) is calculated from the active dataset. If a value hasn't been calculated, the app shows *Unavailable* or *Not evaluated* instead of inventing one.

---

## Features

| Step | What it does |
| --- | --- |
| **Upload** | CSV / TSV / Excel (`.xlsx`, `.xls`) up to 200 MB, with delimiter and encoding detection. Ten built-in samples. Shows a preview, detected column types (numeric / categorical / datetime / text), basic statistics and metadata. Original and cleaned data can be downloaded. |
| **Data Health** | Completeness, uniqueness, rule-based validity and consistency, an optional documented health score, missing-value charts, duplicates, cardinality, constant and identifier-like columns, Tukey outliers, and target class distribution. |
| **Clean Data** | Mean / median / mode / constant / KNN imputation, dropping rows or sparse columns, duplicate removal, type conversion, date parsing, category normalisation, text cleanup, outlier capping or removal, scaling, encoding, renaming and removal. You preview each step before applying it. Every step is logged and can be undone or reset. |
| **EDA** | Interactive Plotly charts: histograms with KDE, box and violin plots, category counts, scatter plots and scatter matrices, grouped aggregations, Pearson/Spearman heatmaps, missing-value matrices, outlier shares, time series, and target relationships. Each chart has a written interpretation built from the computed values. |
| **Insights** | Rule-based findings structured as *Observation → Evidence → Interpretation → Limitation → Next step*, plus a data-story narrative. |
| **Modeling** | Classification (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN) and regression (Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting). Includes automatic task detection with manual override, cross-validation, grid or randomized search, class weights, and SMOTE. |
| **Results** | A leaderboard, per-model metrics, confusion matrix, ROC curve, per-class report, predicted-vs-actual and residual plots, and **held-out permutation importance**. |
| **Agent** | A dataset assistant that answers from the computed analysis and lists its sources. It works without an API key. |
| **AI Report** | PDF, Markdown and HTML exports with the same sections and values as the app. |

## Architecture

```text
app.py                 entry point: page setup, navigation, error boundary
views/                 one file per workflow step (Streamlit UI only)
components/            styling (DataStory theme), navigation stepper, shared UI helpers
core/                  analysis engine — no Streamlit imports, fully unit-tested
  data_loader.py       uploads, sample registry, synthetic-data labels
  profiler.py          column typing, statistics, missing values, duplicates, outliers, correlations
  quality_metrics.py   completeness, uniqueness, rule-based validity/consistency, health score
  cleaning.py          pure cleaning operations, preview records, history replay
  eda.py               Plotly figure builders and evidence-based interpretations
  modeling.py          leakage-safe pipelines, CV, tuning, imbalance handling, metrics
  explainability.py    held-out permutation importance
  insights.py          deterministic insight engine
  results.py           the shared results structure
  reporting.py         Markdown / HTML / PDF rendering from one set of sections
  state.py             session state + dependency invalidation
ai/                    retrieval, assistant, optional LLM client and prompts
data/sample_datasets/  built-in samples
tests/                 pytest suite (unit, credibility and end-to-end page tests)
```

**One source of truth.** `core/results.py` builds a single results dictionary (profile, quality, correlations, cleaning history, model results, importance, insights). The dashboard, the insight engine, the assistant and all three report formats read from it, so they cannot disagree.

**State and invalidation.** `core/state.py` tracks the chain *dataset → cleaning → target/rules → models → importance*. Loading a new dataset clears everything. Applying, undoing or resetting a cleaning step clears trained models and importance. Changing the target or rules rebuilds the results. Models train only when you click **Train models**; they never retrain on a rerun.

## Installation & local run

Python 3.10+ (3.12 recommended).

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io) choose **New app**, select the repository, branch `main`, and main file **`app.py`**.
3. Optional: under **Advanced settings → Secrets**, add an LLM key (see below).
4. Deploy. No separate backend, database or Node.js build is needed.

`.streamlit/config.toml` sets the theme and a 200 MB upload limit. `runtime.txt` requests Python 3.12.

## Data-quality metrics

| Metric | Definition |
| --- | --- |
| **Completeness** | Non-missing cells ÷ all cells in the included columns. You can exclude columns that are legitimately optional; excluded columns are listed next to the score and in reports. |
| **Uniqueness** | Rows that are not an exact repeat of an earlier row ÷ all rows. All columns are compared, and the first occurrence counts as unique. |
| **Validity** | Values passing **your configured** single-column rules (numeric range, allowed values, parses as number/date) ÷ non-missing values those rules apply to. |
| **Consistency** | Records passing **your configured** cross-field (`A <= B`), format (regex) or letter-case rules ÷ records where every column in the rule is present. |

If no rules are configured, validity and consistency show **"Not evaluated — validation rules not configured."** One dimension is never substituted for another.

**Data health score** is a weighted average of the evaluated dimensions: completeness 40%, uniqueness 20%, validity 20%, consistency 20%. Weights for dimensions that weren't evaluated are redistributed proportionally. The weights are an arbitrary DataStory convention. The score is a convenience summary, not a validated statistical measure, and it says nothing about whether the data is fit for a particular purpose.

**Outliers** use Tukey's rule: a value is flagged if it is below Q1 − 1.5×IQR or above Q3 + 1.5×IQR.

## Modeling methodology

- The **train/test split happens first** (stratified for classification, fixed seed).
- Imputation (median for numeric, most-frequent for categorical), optional standard scaling and one-hot encoding run inside a scikit-learn `ColumnTransformer` within a `Pipeline`. They are fitted on training data only, and **re-fitted inside every cross-validation fold**.
- **SMOTE** is a step of an `imblearn.pipeline.Pipeline`, so it resamples only during `fit`, which means only training folds. Validation folds and the test set are never resampled. Class weighting is used where the estimator supports it.
- Datetime features become year/month/day/day-of-week. Free-text and very high-cardinality columns (over 50 levels) are excluded and listed.
- Classification is scored with accuracy, precision, recall, F1 (binary averaging for two classes, weighted for more) and ROC-AUC. Regression is scored with MAE, MSE, RMSE and R². The two metric sets are never mixed.
- **Best-model selection**: with cross-validation on, models are ranked by mean CV score on the training data and the test set is not used for selection. With CV off, models are ranked on the test set, and the app warns that this score is slightly optimistic.
- Edge cases are handled with readable messages: single-class targets, too few rows, classes with one member, splits that are too small, non-numeric regression targets, and models that fail to train.

## Feature importance (held-out permutation importance)

Feature importance uses `sklearn.inspection.permutation_importance` on the **fitted pipeline** and the **untouched test set**, with the ranking metric as the scorer, a fixed seed, and configurable repeats. The app shows the mean and standard deviation for each original input column.

- It measures how much the test score changes when a feature is shuffled. The values are **in metric units, not percentages**, and they don't sum to 100%.
- **Negative values are shown as computed** and are never filtered out.
- Correlated features can share or mask each other's importance.
- Importance shows what the model relies on. **It does not show what causes the outcome.**
- The app warns about small test sets and about features whose standard deviation exceeds their mean.
- Without a fitted model and evaluation data, the app shows *"Feature importance unavailable — a valid fitted model and evaluation dataset are required."*

SHAP is not included in this version.

## Sample datasets and synthetic-data disclosure

| Sample | Status |
| --- | --- |
| Titanic, Customer Churn, Bike Sharing, Student Performance | **Synthetic Dataset — For Demonstration Purposes.** These are generated values that only borrow the column layout of well-known datasets. They are not real observations. The label appears in the sample picker, the header, every page, the insights, the assistant's overview and every exported report. |
| Iris, Wine, Breast Cancer Wisconsin, Diabetes, Digits, Linnerud | Values match the copies bundled with scikit-learn (`sklearn.datasets`, BSD-3-Clause), which was checked numerically. The original sources are cited in the app. |

## AI assistant and optional LLM

The assistant and insight engine **work fully without any API key**:

- **Insights** are generated by deterministic rules from the shared results.
- The **assistant** uses TF-IDF retrieval over documents generated from the computed analysis, plus a method glossary. It quotes those facts and lists its sources. It says so when something hasn't been computed (for example, "no model has been trained yet") and refuses questions about individual rows or anything outside the computed analysis.

Optionally, an OpenAI-compatible LLM can rephrase answers and write the narrative. Set **one** of these as an environment variable or in `.streamlit/secrets.toml` / Streamlit Cloud secrets:

```toml
OPENROUTER_API_KEY = "..."      # or DATASTORY_LLM_API_KEY / OPENAI_API_KEY
# optional:
DATASTORY_LLM_MODEL = "mistralai/mistral-7b-instruct"
DATASTORY_LLM_BASE_URL = "https://openrouter.ai/api/v1"
```

The LLM only receives verified facts. Before showing a response, the app checks every number in it against those facts. If any figure can't be traced, the LLM's text is discarded and the verified facts are shown instead. Keys are read at runtime and are never logged, displayed or committed. `.streamlit/secrets.toml` and `.env` are git-ignored.

## Reports

**AI Report** exports **PDF** (ReportLab with Unicode fonts, paginated, with charts and a synthetic-data footer when relevant), **Markdown** and **HTML**. All three are rendered from the same section blocks, built from the shared results:

1. Dataset overview
2. Data provenance
3. Data-quality assessment
4. Cleaning operations
5. Exploratory analysis
6. ML methodology
7. Model evaluation
8. Feature importance
9. Analytical narrative
10. Limitations
11. Recommendations

Sections can be deselected. Values that weren't computed appear as *Unavailable*. No business impact or performance improvement is claimed.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite (62 tests) covers ingestion, missing-value and duplicate statistics, completeness, rule-based validity and consistency, cleaning correctness and history replay, train/test separation, preprocessing fitted on training rows only, task-specific metrics, SMOTE placement, tuning, reproducibility, held-out permutation importance, report consistency, synthetic labelling, unavailable metrics, and a headless end-to-end run of every page with Streamlit's `AppTest`.

The credibility checks required for the rebuild are included:

| Check | Test |
| --- | --- |
| Importance is derived from a fitted model and test data | `test_permutation_importance_is_computed_not_hardcoded` |
| Quality metrics respond to data changes | `test_quality_metrics_change_with_input` |
| Unavailable values are marked unavailable | `test_validity_and_consistency_unavailable_without_rules`, `test_unavailable_values_without_model` |
| Synthetic samples are labelled | `test_synthetic_samples_are_labelled`, `test_synthetic_label_reaches_every_output` |
| Reports match the app | `test_reports_contain_the_computed_metrics` |
| No fake metrics in the codebase | `test_no_hardcoded_fake_metrics_in_codebase` |
| Reproducibility | `test_training_is_reproducible` |

## What changed from the previous version

The previous React + Vite frontend and FastAPI + SQLite backend were replaced by a single Streamlit app. Credibility fixes:

- **Removed hard-coded feature importance.** The old report computed it as `0.34 − index × 0.045`. It is replaced by held-out permutation importance.
- **Removed fabricated quality scores**: Consistency 91, Validity 93, and Uniqueness 88/97.
- **Removed the invented "Estimated Lift +3.2%".**
- Replaced the opaque health score with a documented, weighted formula.
- Labelled the four synthetic samples, which had been presented without any disclosure.

## Limitations

- Column-type and identifier detection are heuristics. Use **Convert data type** to correct them.
- Very large files (over 200,000 rows) use a seeded random sample for charts and training. Statistics use every row.
- Model scores come from a single train/test split. Use cross-validation for a steadier estimate.
- KNN imputation and grid search can be slow on large data.
- The assistant answers only from computed summaries. It cannot inspect individual rows.
- Session state lives in memory, so reloading the browser tab starts a fresh session.
- SHAP explanations are not implemented.

## License

See [LICENSE](LICENSE).
