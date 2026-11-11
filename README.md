# DataStory AI

**An evidence-based analytics workspace built with Streamlit.** DataStory AI behaves like a careful junior data analyst:

- It understands a dataset and recommends appropriate analyses.
- It cleans data transparently and charts what matters.
- It trains and explains leakage-safe models and finds segments.
- It answers questions with verified computations and produces reports and runnable notebooks.

Every claim is traceable to a computation on your data. Every transformation is auditable. Nothing is fabricated.

Live app: https://datastory-ai.streamlit.app/

## Features

| Area | What it does |
| --- | --- |
| **Dataset & overview** | Upload CSV/TSV/Excel (≤200 MB, delimiter and encoding detected) or load a sample. A dataset-aware overview follows: snapshot, *What DataStory understands*, uncertainty-aware domain inference (you can correct it), ranked target candidates, potential risks, and recommended analyses that open the right page with the right settings. |
| **Provenance** | Every dataset carries an explicit category: *Synthetic demonstration dataset*, *Verified bundled public dataset*, *User-uploaded — provenance unverified* or *Generated/transformed*. It also records the original filename, SHA-256, load time, licence/source URL (when verified) and the transformation lineage (v0, v1, …). Uploads are never labelled synthetic or authentic, and nothing is inferred from file names. |
| **Data health** | Completeness and uniqueness, plus validity and consistency only when you configure rules. Unevaluated dimensions say *"Not evaluated — validation rules not configured."* |
| **Cleaning** | Thirteen operations, each with **configure → preview → apply**. The preview reports exact impact before you apply anything (details below the table) and highlights only the cells that changed. Includes history, undo, reset and CSV export. |
| **Compare** | Original (v0) vs active version: summary deltas, per-column statistics, distribution overlays, category shifts, missing values per column, and an audit trail with undo/restore. |
| **Explore (EDA)** | **Recommended charts** ranked for this dataset, each with its purpose, prerequisites, limitations and a *Generate chart* action. A **chart builder** turns plain language into an editable, validated chart specification. Manual views cover univariate, bivariate, correlation, missing & outliers, time series and target analysis. Identifiers and near-unique columns are never charted as frequencies. Comma-separated multi-value columns are split, long labels are truncated (full text on hover), and the rest is grouped into top-N plus *Other*. |
| **Insights** | Editorial findings (observation → evidence → interpretation → limitation → next step) with evidence IDs, an evidence register, and an optional LLM-polished data story whose numbers are verified. |
| **Model** | Target suitability assessment: identifier, free-text and >50-class targets are blocked, and discouraged targets need explicit acknowledgement. A readiness check covers leakage, imbalance, split feasibility, sparsity and feature/row ratio, with recommended metric, models, CV and imbalance strategy. Leakage-safe pipelines; advanced settings sit in an expander. |
| **Results** | Leaderboard, classification diagnostics (confusion matrix in counts or normalised, ROC, PR curve/PR-AUC, calibration, probability distributions, per-class metrics, error analysis), regression diagnostics (actual vs predicted, residuals, absolute error, error by subgroup), cross-validation score distributions (labelled as training-fold results), and held-out permutation importance. |
| **Segments** | Clustering without a target: K-Means, DBSCAN, agglomerative and Gaussian mixture, with scaling, one-hot encoding, optional PCA, and an elbow/silhouette helper. Metrics are reported only where defined, with reasons otherwise. Includes a PCA projection (labelled as a reduced view), cluster profiles, evidence-based descriptions (no invented personas), renaming and export with labels. |
| **Agent** | Tool-based copilot with rich answers (text, tables, interactive charts), evidence, follow-ups, navigation, chat export and *new conversation*. |
| **Report** | Report builder with four modes (Executive, EDA, Machine learning, Comprehensive technical). You choose and order sections, select charts, write a title and add commentary that is visibly labelled as user-authored. Preview and export to PDF, Markdown or HTML; a changed analysis is never silently reused. |
| **Notebook** | A runnable `.ipynb` that **re-executes** the analysis: load with a SHA-256 check, profile, every cleaning step as explicit pandas code (with a content-hash check), saved charts, the same model pipelines and seeds (test metrics compared with the app's), permutation importance and clustering. Optional bundle with dataset, pinned `requirements.txt` and README. |
| **Sessions** | Export/import a versioned `.datastory.zip`. It is validated by SHA-256 checksums, schema version and an allow-list of member names, with zip-bomb and path-traversal guards. JSON only (no pickles), and the dataset can be included or left out. Fitted models are not stored; their configuration is restored and they must be retrained. |

### Exact cleaning impact

The preview reports what an operation would actually change, and the same numbers appear after you apply it:

- values flagged, cells examined, values modified, rows and columns affected;
- rows and columns removed or added;
- missing values introduced or resolved.

Comparison is null-safe and dtype-aware, so 2019 and 2019.0 count as equal. Rows keep their index labels through every step, so deleting rows is reported as removals, never as changed cells on shifted rows.

## Architecture

```
app.py                     entry point: theme hand-off, state, CSS, hidden st.navigation, header + flowchart
components/
  styling.py               design tokens (light/dark), typography, flowchart/connector CSS, reduced-motion rules
  theme.py                 light/dark toggle (Streamlit custom component v2) + theme-switch hand-off
  navigation.py            STEPS registry, header with dataset context and Session menu, flowchart navigation
  ui.py                    page headers, callouts, provenance notice, themed Plotly rendering, chart toolbar
core/                      Streamlit-free analytics (unit-tested)
  data_loader.py           ingestion, samples registry, provenance categories, hashes, lineage
  profiler.py              column kinds (incl. month-name dates), profile, identifiers, outliers, correlations
  quality_metrics.py       completeness / uniqueness / rule-based validity & consistency, health score
  cleaning.py, impact.py   pure cleaning operations; exact row-identity-aware change measurement
  comparison.py            original vs cleaned comparison
  charts.py, nl_chart.py   ChartSpec schema, validation, rendering, recommendations; plain language → ChartSpec
  overview.py              domain inference, dataset-aware overview
  targets.py, ml_readiness.py, modeling.py, diagnostics.py, explainability.py
  clustering.py            unsupervised segmentation
  insights.py, results.py  insight engine; shared results structure + evidence register (single source of truth)
  reporting.py, notebook.py, session_io.py, handoff.py, formatting.py, state.py
ai/
  copilot.py               tool registry, argument validation, rule/LLM routing, number guard
  assistant.py, retrieval.py, llm.py, prompts.py
views/                     one file per page (dataset, quality, cleaning, compare, eda, insights, ml,
                           model_results, clustering, assistant, reports)
```

**Shared results.** `core/results.py` builds one structure for the active dataset version: dataset metadata and lineage, profile, quality, cleaning history, charts, model runs, importance, clustering, insights and the evidence register. The dashboard, copilot, reports and notebook all read from it, and it is rebuilt whenever its inputs change. Stale artefacts are flagged rather than reused: trained models are cleared when the data changes, reports are marked out of date, chat answers show the version they were computed on, and clustering is tied to a data version.

**Numbers.** `core/formatting.py` formats every percentage and number. Tiny non-zero shares show as `<0.1%` (never `0.0%`), missing values show as *Unavailable*, and denominators are stated in the text.

## Design system

- **Typography:**
  - Plus Jakarta Sans for the interface and body text;
  - Manrope for display text: headings, page titles and metric values;
  - DM Serif Display only for the italic accents in the landing headline.

  Statistics use tabular figures. Fonts are set in `.streamlit/config.toml`, so Streamlit's widgets, tables and charts use the same families.
- **Themes:** `[theme.light]` and `[theme.dark]` in `config.toml` style every native widget. `components/styling.py` mirrors the same values as CSS tokens (`--ds-bg`, `--ds-surface`, `--ds-accent`, `--ds-line`, `--ds-glow`, …), so colours are defined in one place. Plotly figures are re-themed at render time.
  - Light mode keeps the warm-ivory and sage identity. Dark mode uses deep charcoal surfaces, desaturated sage accents and warm off-white text.
  - The sun/moon toggle in the header stores the choice in the key Streamlit itself reads and reloads the page.
  - The reload starts a new Streamlit session, so the current analysis is parked in server memory under a one-time random token and claimed straight back. The token expires after 2 minutes and nothing is written to disk.
- **Flowchart navigation:**
  - The workflow is a connected chain of native `st.page_link` nodes: Dataset → Health → Clean → Explore → Insights → Model → Results → Agent → Report, with Compare and Segments as optional branches.
  - Each node sits in a keyed container whose key encodes its state (`active`, `done`, `todo`, `locked`). Connectors, arrowheads and the travelling highlight on completed or active links are pure CSS on those containers. Nothing manipulates Streamlit's internal DOM, and clicks always navigate through Streamlit.
  - A stage is complete only when its operation has run on the current data (for example, a chart generated or a model trained). Visiting a page never completes it. Locked nodes explain their prerequisite in a tooltip.
  - Below 860 px wide, the flowchart collapses into a compact *Step n of N* selector.
  - `prefers-reduced-motion` disables all animation, and focus outlines are visible.

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
3. Optional: under **Advanced settings → Secrets**, add `GROQ_API_KEY = "gsk_..."` (see below).
   The app runs entirely in Streamlit; sessions, notebooks and reports are generated in memory and handed to the browser as downloads.
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

## AI copilot and optional LLM

The copilot and insight engine **work fully without any API key**.

- **Insights** are generated by deterministic rules from the shared results. Each one gets an evidence ID (F1, F2, …) that links it to the computation, columns and dataset version behind it.
- The **copilot is tool-based.** Every question is mapped to one of a fixed set of analysis tools, each with a JSON schema:
  - dataset summary, quality issues, column statistics, missing values by column;
  - grouped aggregation, row counts with conditions, correlations, target associations (|r|, correlation ratio η, Cramér's V);
  - natural-language charts, model summary, confusion matrix, feature importance, cleaning suggestions, segments and findings.
- Arguments are validated before anything runs: types, allowed values, required fields, no extra fields, and columns that actually exist. The tool then runs deterministic pandas code on the active session's data and returns text, tables and/or a chart.
- Requests to run code, SQL, shell commands, or to touch files or the network are refused. Generated code is never executed.
- Questions outside the tools fall back to retrieval over the computed analysis and a method glossary.

Optionally, a [Groq](https://console.groq.com) model can **choose the tool** (OpenAI-style function calling), **rephrase tool results** and write the narrative. An LLM-proposed call goes through the same validation. An invalid call is ignored and the rule-based router is used instead. Groq has a free tier. Create a key at https://console.groq.com/keys, then add it as an environment variable or in `.streamlit/secrets.toml` / Streamlit Cloud **Settings → Secrets**:

```toml
GROQ_API_KEY = "gsk_your-key-here"
# optional — any Groq chat model ID (default: openai/gpt-oss-120b)
GROQ_MODEL = "openai/gpt-oss-120b"
```

When a key is set, the Insights and Agent pages show **"LLM connected · <model>"**. Free-tier rate limits apply; if one is hit, the app shows the verified rule-based answer and says why.

The LLM only receives verified facts. Before showing a response, the app checks every number in it against those facts. If any figure can't be traced, the LLM's text is discarded and the verified facts are shown instead. Keys are read at runtime and are never logged, displayed or committed. `.streamlit/secrets.toml` and `.env` are git-ignored.

## Reports and notebooks

The **Report** step is a builder with four modes:

| Mode | Default sections |
| --- | --- |
| Executive report | Overview, provenance, narrative and findings, saved charts, recommendations, limitations |
| Exploratory data analysis report | Overview, provenance, quality, cleaning, EDA, saved charts, limitations, evidence register |
| Machine learning report | Overview, provenance, cleaning, methodology, evaluation, feature importance, limitations, reproducibility |
| Comprehensive technical report | Every section: overview, provenance, quality, cleaning, EDA, saved charts, methodology, evaluation, importance, clustering, narrative, limitations, recommendations, evidence register, reproducibility |

You can include or exclude sections, reorder them, pick which saved charts appear, set the title, and write an opening summary or per-section notes.

- Your text is always printed as *"Analyst commentary (user-authored, not computed)"*.
- Saved charts are stored as specifications and recomputed from the current data when the report is built.
- PDF (ReportLab), Markdown and HTML are rendered from the same blocks, so they cannot disagree.
- If the analysis changes after a report is generated, the report is marked out of date and is never silently reused.

The **Reproducible notebook** tab exports a `.ipynb` (or a bundle with the dataset and pinned requirements) that imports only pandas, numpy, scikit-learn, imbalanced-learn, scipy and plotly. It re-runs:

- the data load, with a SHA-256 check;
- the profile;
- every cleaning step, followed by a content-hash check against the app;
- the saved charts;
- training with the same pipelines, grids, folds and seed, with test metrics compared to the app's;
- permutation importance and clustering.

## Sessions and privacy

- Uploaded data lives only in the memory of your browser tab's Streamlit session. Nothing is written to the server's disk, and cached comparisons are keyed by a random per-load dataset ID.
- **Session → Prepare session package** downloads a `.datastory.zip` (manifest + `state.json` + optionally `data/original.csv`). **Dataset → Resume a saved analysis** validates and restores it.
- Fitted models, permutation importance and anything that would need pickling are **not** stored. Their configuration is restored, and the app tells you to retrain.
- Switching between light and dark mode reloads the page. Your analysis is carried across in server memory for at most two minutes under a one-time random token.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite has **210 tests** (about 2 minutes). It covers:

- ingestion and provenance;
- exact cleaning impact (imputation, outlier capping, row and column deletion, duplicates, type conversion, text normalisation, no-ops, missing↔value transitions, preview = applied = replayed);
- quality metrics;
- target suitability and readiness;
- train/test isolation, SMOTE placement, CV scores and diagnostics;
- permutation importance and clustering;
- chart-spec validation and aggregation correctness, recommendations (identifiers excluded), natural-language chart parsing;
- copilot routing, argument validation, code-execution refusal, number guard and LLM proposals (mocked);
- report builder and evidence register;
- session packages (tampering, path traversal, foreign files, zip bombs, wrong dataset);
- notebook generation and **clean-kernel execution**;
- the theme hand-off;
- flowchart navigation (real links, data-driven completion, locked prerequisites);
- headless end-to-end runs of every page with Streamlit's `AppTest`.

`tests/test_scenarios.py` runs the five acceptance scenarios from the upgrade specification:

| Scenario | Dataset |
| --- | --- |
| A | Netflix-shaped catalogue (the real file isn't bundled; `tests/fixtures.py` generates the same layout) |
| B | Customer churn classification |
| C | Bike-sharing regression, with notebook execution |
| D | Unsupervised segmentation |
| E | Session export and restore |

The credibility checks from the original rebuild are still included (`tests/test_reports_and_credibility.py`): importance comes from a fitted model and held-out data; quality metrics respond to the data; unavailable values say so; synthetic samples are labelled everywhere; reports match the app; no hard-coded metrics exist in the codebase.

## Limitations

- Column-type, identifier and domain detection are heuristics. You can override types (Clean), the domain (Dataset overview) and the target (Model).
- Datasets over 200,000 rows use a seeded sample for charts, training and copilot computations; profile statistics use every row. The notebook trains on all rows, so its metrics differ in that case.
- Model scores come from a single train/test split; use cross-validation for a steadier estimate. No significance tests are run.
- The natural-language chart parser is rule-based. It supports common phrasings and says what it assumed or couldn't find. Ambiguous requests get a stated default or a clarification request.
- Copilot answers are limited to its tools and the computed analysis; it does not browse individual rows.
- Group-aware and temporal splits are flagged by the readiness check but not yet offered as split strategies.
- SHAP and UMAP are not implemented; permutation importance and PCA are the supported methods.
- Switching the theme reloads the page. If the reload takes longer than two minutes, the analysis is not carried over.

## License

See [LICENSE](LICENSE).
