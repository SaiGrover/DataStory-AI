import pandas as pd
import numpy as np
from typing import Dict, Any


AVAILABLE_ACTIONS = [
    "remove_duplicates",
    "drop_empty_rows",
    "drop_high_missing",
    "remove_constant",
    "fill_numeric_median",
    "fill_categorical_mode",
    "strip_whitespace",
    "convert_dates",
    "cap_outliers",
]


def _legacy_actions_to_config(actions, custom_fill_value=None, drop_missing_threshold=0.5) -> Dict[str, Any]:
    actions = actions or []
    return {
        "remove_dups": "remove_duplicates" in actions,
        "drop_empty_rows": "drop_empty_rows" in actions,
        "drop_high_missing": "drop_high_missing" in actions,
        "drop_missing_threshold": drop_missing_threshold,
        "remove_constant": "remove_constant" in actions,
        "numeric_fill": "Custom value" if custom_fill_value is not None else (
            "Median" if "fill_numeric_median" in actions else "None"
        ),
        "custom_num_val": custom_fill_value,
        "cat_fill": "Custom value" if custom_fill_value is not None else (
            "Mode" if "fill_categorical_mode" in actions else "None"
        ),
        "custom_cat_val": custom_fill_value,
        "convert_dates": "convert_dates" in actions,
        "strip_whitespace": "strip_whitespace" in actions,
        "outlier_action": "Cap at IQR bounds (1.5x)" if "cap_outliers" in actions else "None",
    }


def preview_cleaning(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a preview of what cleaning will produce without modifying the original."""
    if not isinstance(config, dict):
        config = _legacy_actions_to_config(config)

    actions = []
    preview_df = df.copy()

    duplicate_subset = [c for c in config.get("duplicate_key_columns", []) if c in preview_df.columns]
    if config.get("remove_dups") or duplicate_subset:
        n_before = len(preview_df)
        preview_df = preview_df.drop_duplicates(subset=duplicate_subset or None)
        removed = n_before - len(preview_df)
        if removed > 0:
            scope = f" using {len(duplicate_subset)} key column(s)" if duplicate_subset else ""
            actions.append(f"Remove {removed:,} duplicate rows{scope}")

    if config.get("drop_empty_rows"):
        n_before = len(preview_df)
        preview_df = preview_df.dropna(how="all")
        removed = n_before - len(preview_df)
        if removed > 0:
            actions.append(f"Drop {removed:,} completely empty rows")

    drop_cols = [c for c in config.get("drop_columns", []) if c in preview_df.columns]
    if drop_cols:
        preview_df = preview_df.drop(columns=drop_cols)
        actions.append(f"Drop selected column(s): {', '.join(drop_cols[:5])}")

    if config.get("drop_duplicate_columns"):
        duplicate_cols = preview_df.columns[preview_df.T.duplicated()].tolist()
        if duplicate_cols:
            preview_df = preview_df.drop(columns=duplicate_cols)
            actions.append(f"Drop {len(duplicate_cols)} duplicate column(s): {', '.join(duplicate_cols[:3])}")

    if config.get("drop_high_missing"):
        threshold = float(config.get("drop_missing_threshold", 0.5))
        threshold = min(max(threshold, 0), 1)
        high = [c for c in preview_df.columns if preview_df[c].isnull().mean() > threshold]
        if high:
            preview_df = preview_df.drop(columns=high)
            actions.append(f"Drop {len(high)} column(s) with >{int(threshold * 100)}% missing: {', '.join(high[:3])}")

    if config.get("remove_constant"):
        const = [c for c in preview_df.columns if preview_df[c].nunique() <= 1]
        if const:
            preview_df = preview_df.drop(columns=const)
            actions.append(f"Remove {len(const)} constant column(s): {', '.join(const[:3])}")

    num_cols = preview_df.select_dtypes(include="number").columns.tolist()
    selected_num_cols = [c for c in config.get("numeric_columns", []) if c in num_cols]
    if selected_num_cols:
        num_cols = selected_num_cols
    cat_cols = preview_df.select_dtypes(include=["object", "category"]).columns.tolist()
    selected_cat_cols = [c for c in config.get("categorical_columns", []) if c in cat_cols]
    fill_cat_cols = selected_cat_cols or cat_cols
    text_cols = [c for c in config.get("text_columns", []) if c in cat_cols] or cat_cols
    outlier_cols = [c for c in config.get("outlier_columns", []) if c in num_cols] or num_cols

    fill_strategy = config.get("numeric_fill", "None")
    if fill_strategy == "Mean" and num_cols:
        preview_df[num_cols] = preview_df[num_cols].fillna(preview_df[num_cols].mean())
        actions.append(f"Fill numeric missing values with column mean ({len(num_cols)} cols)")
    elif fill_strategy == "Median" and num_cols:
        preview_df[num_cols] = preview_df[num_cols].fillna(preview_df[num_cols].median())
        actions.append(f"Fill numeric missing values with column median ({len(num_cols)} cols)")
    elif fill_strategy == "Custom value" and config.get("custom_num_val") is not None:
        preview_df[num_cols] = preview_df[num_cols].fillna(config["custom_num_val"])
        actions.append(f"Fill numeric missing values with {config['custom_num_val']}")

    cat_strategy = config.get("cat_fill", "None")
    if cat_strategy == "Mode" and fill_cat_cols:
        for col in fill_cat_cols:
            if preview_df[col].isnull().any():
                mode_val = preview_df[col].mode()
                if len(mode_val) > 0:
                    preview_df[col] = preview_df[col].fillna(mode_val.iloc[0])
        actions.append(f"Fill categorical missing values with mode ({len(fill_cat_cols)} cols)")
    elif cat_strategy == "Custom value" and config.get("custom_cat_val") and fill_cat_cols:
        preview_df[fill_cat_cols] = preview_df[fill_cat_cols].fillna(config["custom_cat_val"])
        actions.append(f"Fill categorical missing values with '{config['custom_cat_val']}'")

    if config.get("strip_whitespace") and text_cols:
        for col in text_cols:
            if preview_df[col].dtype == object:
                preview_df[col] = preview_df[col].str.strip()
        actions.append(f"Strip whitespace from text columns ({len(text_cols)} cols)")

    text_case = config.get("text_case", "Keep as-is")
    if text_case != "Keep as-is" and text_cols:
        for col in text_cols:
            if preview_df[col].dtype == object:
                if text_case == "Lowercase":
                    preview_df[col] = preview_df[col].str.lower()
                elif text_case == "Uppercase":
                    preview_df[col] = preview_df[col].str.upper()
                elif text_case == "Title Case":
                    preview_df[col] = preview_df[col].str.title()
        actions.append(f"Normalize text case to {text_case.lower()} ({len(text_cols)} cols)")

    type_cols = [c for c in config.get("type_columns", []) if c in preview_df.columns]
    coerce_cols = [c for c in (type_cols or cat_cols) if c in preview_df.columns]
    if config.get("coerce_numeric_text") and coerce_cols:
        converted = []
        for col in coerce_cols:
            cleaned = preview_df[col].astype(str).str.replace(",", "", regex=False).str.strip()
            parsed = pd.to_numeric(cleaned, errors="coerce")
            if parsed.notna().mean() >= 0.8:
                preview_df[col] = parsed
                converted.append(col)
        if converted:
            actions.append(f"Convert numeric-looking text to numbers: {', '.join(converted[:5])}")

    date_cols = [c for c in config.get("date_columns", []) if c in preview_df.columns]
    convert_date_cols = [c for c in (date_cols or cat_cols) if c in preview_df.columns]
    if config.get("convert_dates") and convert_date_cols:
        converted_dates = []
        for col in convert_date_cols:
            try:
                parsed = pd.to_datetime(preview_df[col], errors="coerce")
                if parsed.notna().mean() > 0.7:
                    preview_df[col] = parsed
                    converted_dates.append(col)
            except Exception:
                pass
        if converted_dates:
            actions.append(f"Convert possible date columns: {', '.join(converted_dates[:5])}")

    outlier_action = config.get("outlier_action", "None")
    if outlier_action.startswith("Cap at IQR bounds"):
        for col in outlier_cols:
            q1, q3 = preview_df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            preview_df[col] = preview_df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        actions.append(f"Cap outliers at IQR bounds (1.5x) for {len(outlier_cols)} numeric columns")
    elif outlier_action == "Remove outlier rows":
        mask = pd.Series([True] * len(preview_df))
        for col in outlier_cols:
            q1, q3 = preview_df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            mask &= (preview_df[col] >= q1 - 1.5 * iqr) & (preview_df[col] <= q3 + 1.5 * iqr)
        n_removed = (~mask).sum()
        if n_removed > 0:
            actions.append(f"Remove {n_removed:,} outlier rows")

    if not actions:
        actions.append("No cleaning actions selected - dataset unchanged.")

    return {
        "rows": len(preview_df),
        "columns": len(preview_df.columns),
        "missing": int(preview_df.isnull().sum().sum()),
        "duplicates": int(preview_df.duplicated().sum()),
        "actions": actions,
    }


def apply_cleaning(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Apply cleaning config and return the cleaned dataframe."""
    if not isinstance(config, dict):
        config = _legacy_actions_to_config(config)

    df = df.copy()

    duplicate_subset = [c for c in config.get("duplicate_key_columns", []) if c in df.columns]
    if config.get("remove_dups") or duplicate_subset:
        df = df.drop_duplicates(subset=duplicate_subset or None)

    if config.get("drop_empty_rows"):
        df = df.dropna(how="all")

    drop_cols = [c for c in config.get("drop_columns", []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if config.get("drop_duplicate_columns"):
        duplicate_cols = df.columns[df.T.duplicated()].tolist()
        if duplicate_cols:
            df = df.drop(columns=duplicate_cols)

    if config.get("drop_high_missing"):
        threshold = float(config.get("drop_missing_threshold", 0.5))
        threshold = min(max(threshold, 0), 1)
        high = [c for c in df.columns if df[c].isnull().mean() > threshold]
        if high:
            df = df.drop(columns=high)

    if config.get("remove_constant"):
        const = [c for c in df.columns if df[c].nunique() <= 1]
        if const:
            df = df.drop(columns=const)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    selected_num_cols = [c for c in config.get("numeric_columns", []) if c in num_cols]
    if selected_num_cols:
        num_cols = selected_num_cols
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    selected_cat_cols = [c for c in config.get("categorical_columns", []) if c in cat_cols]
    fill_cat_cols = selected_cat_cols or cat_cols
    text_cols = [c for c in config.get("text_columns", []) if c in cat_cols] or cat_cols

    fill_strategy = config.get("numeric_fill", "None")
    if fill_strategy == "Mean" and num_cols:
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif fill_strategy == "Median" and num_cols:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif fill_strategy == "Custom value" and config.get("custom_num_val") is not None:
        df[num_cols] = df[num_cols].fillna(config["custom_num_val"])

    cat_strategy = config.get("cat_fill", "None")
    if cat_strategy == "Mode" and fill_cat_cols:
        for col in fill_cat_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val.iloc[0])
    elif cat_strategy == "Custom value" and config.get("custom_cat_val") and fill_cat_cols:
        df[fill_cat_cols] = df[fill_cat_cols].fillna(config["custom_cat_val"])

    if config.get("strip_whitespace") and text_cols:
        for col in text_cols:
            if df[col].dtype == object:
                df[col] = df[col].str.strip()

    text_case = config.get("text_case", "Keep as-is")
    if text_case != "Keep as-is" and text_cols:
        for col in text_cols:
            if df[col].dtype == object:
                if text_case == "Lowercase":
                    df[col] = df[col].str.lower()
                elif text_case == "Uppercase":
                    df[col] = df[col].str.upper()
                elif text_case == "Title Case":
                    df[col] = df[col].str.title()

    type_cols = [c for c in config.get("type_columns", []) if c in df.columns]
    coerce_cols = [c for c in (type_cols or cat_cols) if c in df.columns]
    if config.get("coerce_numeric_text") and coerce_cols:
        for col in coerce_cols:
            cleaned = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
            parsed = pd.to_numeric(cleaned, errors="coerce")
            if parsed.notna().mean() >= 0.8:
                df[col] = parsed

    if config.get("convert_dates"):
        date_cols = [c for c in config.get("date_columns", []) if c in df.columns]
        convert_date_cols = [c for c in (date_cols or cat_cols) if c in df.columns]
        for col in convert_date_cols:
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if parsed.notna().mean() > 0.7:
                    df[col] = parsed
            except Exception:
                pass

    outlier_action = config.get("outlier_action", "None")
    num_cols_current = df.select_dtypes(include="number").columns.tolist()
    selected_outlier_cols = [c for c in config.get("outlier_columns", []) if c in num_cols_current]
    if selected_outlier_cols:
        num_cols_current = selected_outlier_cols
    if outlier_action.startswith("Cap at IQR bounds"):
        for col in num_cols_current:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    elif outlier_action == "Remove outlier rows":
        mask = pd.Series([True] * len(df), index=df.index)
        for col in num_cols_current:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            mask &= (df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)
        df = df[mask]

    df = df.reset_index(drop=True)
    return df


def clean_dataset(df: pd.DataFrame, actions, custom_fill_value=None, drop_missing_threshold=0.5):
    """Compatibility wrapper for the FastAPI router layer."""
    config = _legacy_actions_to_config(actions, custom_fill_value, drop_missing_threshold)
    before = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }
    preview = preview_cleaning(df, config)
    cleaned = apply_cleaning(df, config)
    after = {
        "rows": len(cleaned),
        "columns": len(cleaned.columns),
        "missing": int(cleaned.isnull().sum().sum()),
        "duplicates": int(cleaned.duplicated().sum()),
    }
    return cleaned, preview.get("actions", []), {"before": before, "after": after}
