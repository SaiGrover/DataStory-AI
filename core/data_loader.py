"""Dataset ingestion: uploads (CSV / Excel) and the built-in sample registry.

Nothing here depends on Streamlit, so it can be tested directly.
"""
from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_datasets"

SYNTHETIC_LABEL = "Synthetic Dataset — For Demonstration Purposes"

MAX_UPLOAD_MB = 200
MAX_ROWS_FULL_ANALYSIS = 200_000  # above this, profiling charts work on a seeded sample
ALLOWED_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xls")

_SKLEARN_NOTE = (
    "Values match the copy bundled with scikit-learn (sklearn.datasets), "
    "distributed under scikit-learn's BSD-3-Clause license. Original source: {origin}."
)


class DataLoadError(ValueError):
    """Raised with a user-readable message when a dataset cannot be loaded."""


@dataclass
class DatasetMeta:
    name: str
    source: str  # "upload" | "sample"
    synthetic: bool = False
    provenance: str = "Uploaded by the user. Provenance not verified by DataStory AI."
    description: str = ""
    suggested_target: str | None = None
    file_size_kb: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str | None:
        return SYNTHETIC_LABEL if self.synthetic else None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["label"] = self.label
        return d


@dataclass(frozen=True)
class Sample:
    id: str
    name: str
    file: str
    description: str
    synthetic: bool
    provenance: str
    target: str | None


_SYN = (
    "Generated artificially for this demo. It only borrows the column layout of the "
    "well-known '{orig}' dataset; the values are not real observations and must not be "
    "cited as findings about the real world."
)

SAMPLES: dict[str, Sample] = {
    s.id: s
    for s in [
        Sample("titanic", "Titanic", "titanic.csv", "Passenger survival (classification)", True,
               _SYN.format(orig="Titanic passenger list"), "Survived"),
        Sample("churn", "Customer Churn", "customer_churn.csv", "Telecom churn (classification)", True,
               _SYN.format(orig="Telco customer churn"), "Churn"),
        Sample("bike", "Bike Sharing", "bike_sharing.csv", "Daily rental counts (regression)", True,
               _SYN.format(orig="Capital Bikeshare daily"), "cnt"),
        Sample("student", "Student Performance", "student_performance.csv", "Final grade G3 (regression)", True,
               _SYN.format(orig="UCI Student Performance"), "G3"),
        Sample("iris", "Iris", "iris.csv", "Flower species (classification)", False,
               _SKLEARN_NOTE.format(origin="R. A. Fisher (1936), UCI Machine Learning Repository"), "species"),
        Sample("wine", "Wine", "wine.csv", "Cultivar from chemistry (classification)", False,
               _SKLEARN_NOTE.format(origin="Forina et al., UCI Machine Learning Repository"), "target"),
        Sample("breast_cancer", "Breast Cancer Wisconsin", "breast_cancer.csv", "Diagnosis (classification)", False,
               _SKLEARN_NOTE.format(origin="Wolberg, Street & Mangasarian, UCI Machine Learning Repository"), "target"),
        Sample("diabetes", "Diabetes", "diabetes.csv", "Disease progression (regression)", False,
               _SKLEARN_NOTE.format(origin="Efron et al. (2004), 'Least Angle Regression'"), "target"),
        Sample("digits", "Digits", "digits.csv", "8×8 handwritten digits (classification)", False,
               _SKLEARN_NOTE.format(origin="Alpaydin & Kaynak, UCI Optical Recognition of Handwritten Digits"), "target"),
        Sample("linnerud", "Linnerud", "linnerud.csv", "Exercise vs. physiology (regression, 20 rows)", False,
               _SKLEARN_NOTE.format(origin="Tenenhaus (1998), 'La régression PLS'"), "Weight"),
    ]
}


def sample_display_name(sample_id: str) -> str:
    """Selector label. Synthetic samples always carry the disclosure."""
    s = SAMPLES[sample_id]
    return f"{s.name} · {SYNTHETIC_LABEL}" if s.synthetic else s.name


def load_sample(sample_id: str) -> tuple[pd.DataFrame, DatasetMeta]:
    if sample_id not in SAMPLES:
        raise DataLoadError(f"Unknown sample dataset '{sample_id}'.")
    s = SAMPLES[sample_id]
    path = SAMPLES_DIR / s.file
    if not path.exists():
        raise DataLoadError(f"Sample file '{s.file}' is missing from data/sample_datasets.")
    df = _finalise(pd.read_csv(path))
    meta = DatasetMeta(
        name=s.name, source="sample", synthetic=s.synthetic, provenance=s.provenance,
        description=s.description, suggested_target=s.target,
        file_size_kb=round(path.stat().st_size / 1024, 1),
    )
    return df, meta


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _read_delimited(data: bytes) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text_head = data[:65536].decode(enc)
        except UnicodeDecodeError:
            continue
        sep = _sniff_delimiter(text_head)
        try:
            df = pd.read_csv(io.BytesIO(data), sep=sep, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise DataLoadError("The file contains no data.") from exc
        except pd.errors.ParserError as exc:
            raise DataLoadError(f"The file could not be parsed as delimited text: {exc}") from exc
        if enc not in ("utf-8-sig", "utf-8"):
            notes.append(f"Decoded with fallback encoding {enc}.")
        if sep != ",":
            notes.append(f"Detected delimiter {sep!r}.")
        return df, notes
    raise DataLoadError("The file's text encoding could not be determined.")


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column labels: strings, stripped, unique."""
    cols, seen = [], {}
    for i, c in enumerate(df.columns):
        name = str(c).strip() or f"column_{i + 1}"
        if name.startswith("Unnamed:"):
            name = f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        cols.append(name)
    df = df.copy()
    df.columns = cols
    return df.reset_index(drop=True)


def load_upload(filename: str, data: bytes) -> tuple[pd.DataFrame, DatasetMeta]:
    """Parse an uploaded file. Raises DataLoadError with a readable message."""
    name = Path(filename).name
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise DataLoadError(
            f"Unsupported file type '{ext or name}'. Upload a CSV, TSV, or Excel (.xlsx / .xls) file."
        )
    if not data:
        raise DataLoadError("The uploaded file is empty (0 bytes).")
    size_mb = len(data) / 1024 / 1024
    if size_mb > MAX_UPLOAD_MB:
        raise DataLoadError(f"The file is {size_mb:.0f} MB; the limit is {MAX_UPLOAD_MB} MB.")

    notes: list[str] = []
    if ext in (".xlsx", ".xls"):
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        except ImportError as exc:
            pkg = "openpyxl" if ext == ".xlsx" else "xlrd"
            raise DataLoadError(f"Reading {ext} files requires the '{pkg}' package, which is not installed.") from exc
        except Exception as exc:  # corrupt workbook, password protected, ...
            raise DataLoadError(f"The Excel file could not be read: {exc}") from exc
        non_empty = {k: v for k, v in sheets.items() if not v.empty}
        if not non_empty:
            raise DataLoadError("The workbook has no sheet containing data.")
        sheet, df = next(iter(non_empty.items()))
        if len(sheets) > 1:
            notes.append(f"Workbook has {len(sheets)} sheets; loaded the first non-empty one ('{sheet}').")
    else:
        df, notes = _read_delimited(data)

    if df.shape[1] == 0:
        raise DataLoadError("No columns were found in the file.")
    if df.shape[0] == 0:
        raise DataLoadError("The file has a header row but no data rows.")
    all_empty = [c for c in df.columns if df[c].isna().all()]
    if len(all_empty) == df.shape[1]:
        raise DataLoadError("Every cell in the file is empty.")

    df = _finalise(df)
    if len(df) > MAX_ROWS_FULL_ANALYSIS:
        notes.append(
            f"Large dataset ({len(df):,} rows): statistics use all rows; charts and model "
            f"training use a seeded random sample of {MAX_ROWS_FULL_ANALYSIS:,} rows."
        )
    meta = DatasetMeta(name=name, source="upload", file_size_kb=round(len(data) / 1024, 1), notes=notes)
    return df, meta


def sample_for_display(df: pd.DataFrame, limit: int = MAX_ROWS_FULL_ANALYSIS, seed: int = 42) -> pd.DataFrame:
    """Seeded down-sample used for charts / training on very large frames."""
    return df if len(df) <= limit else df.sample(limit, random_state=seed).sort_index()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
