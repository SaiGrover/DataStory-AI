import gc
import os

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Regression models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)

from backend.services.openrouter import generate_explanation


CLASSIFICATION_GRIDS = {
    "Logistic Regression": {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs"],
        "clf__max_iter": [500],
    },
    "Decision Tree Classifier": {
        "clf__max_depth": [5, 10],
        "clf__min_samples_split": [2, 10],
        "clf__min_samples_leaf": [1, 4],
        "clf__criterion": ["gini"],
    },
    "Random Forest Classifier": {
        "clf__n_estimators": [80],
        "clf__max_depth": [8, None],
        "clf__min_samples_split": [2],
        "clf__min_samples_leaf": [1, 2],
    },
    "Support Vector Machine": {
        "clf__C": [0.5, 1.0],
        "clf__kernel": ["rbf"],
        "clf__gamma": ["scale"],
    },
    "K-Nearest Neighbors": {
        "clf__n_neighbors": [3, 7, 11],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean"],
    },
    "Naive Bayes": {
        "clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
    },
    "Gradient Boosting Classifier": {
        "clf__n_estimators": [50, 100],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [2],
    },
    "Extra Trees Classifier": {
        "clf__n_estimators": [80],
        "clf__max_depth": [8, None],
        "clf__min_samples_leaf": [1, 2],
    },
}

REGRESSION_GRIDS = {
    "Linear Regression": {
        "clf__fit_intercept": [True, False],
    },
    "Ridge Regression": {
        "clf__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "Decision Tree Regressor": {
        "clf__max_depth": [5, 10],
        "clf__min_samples_split": [2, 10],
        "clf__min_samples_leaf": [1, 4],
    },
    "Random Forest Regressor": {
        "clf__n_estimators": [80],
        "clf__max_depth": [8, None],
        "clf__min_samples_split": [2],
        "clf__min_samples_leaf": [1, 2],
    },
    "Gradient Boosting Regressor": {
        "clf__n_estimators": [60, 100],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [2],
    },
    "Extra Trees Regressor": {
        "clf__n_estimators": [80],
        "clf__max_depth": [8, None],
        "clf__min_samples_leaf": [1, 2],
    },
    "Support Vector Regressor": {
        "clf__C": [0.5, 1.0],
        "clf__epsilon": [0.05, 0.1],
        "clf__kernel": ["rbf"],
    },
}

NEEDS_SCALING = {"Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors", "Support Vector Regressor"}
SUPPORTS_CLASS_WEIGHT = {
    "Logistic Regression", "Decision Tree Classifier",
    "Random Forest Classifier", "Support Vector Machine", "Extra Trees Classifier",
}
TRAIN_JOBS = max(1, int(os.getenv("DATASTORY_TRAIN_JOBS", "1")))
MAX_CV_FOLDS = max(2, int(os.getenv("DATASTORY_MAX_CV_FOLDS", "3")))
DENSE_PREPROCESSING_MODELS = {"Naive Bayes"}


def get_recommended_models(task_type: str) -> List[str]:
    grids = CLASSIFICATION_GRIDS if task_type == "classification" else REGRESSION_GRIDS
    return list(grids.keys())


def detect_task_type(target: pd.Series):
    """Compatibility task detector used by the older FastAPI router."""
    warnings = []
    nuniq = target.nunique(dropna=True)
    if target.dtype == object or target.dtype.name == "category" or target.dtype == bool:
        task = "classification"
    elif nuniq <= 20:
        task = "classification"
        warnings.append("Numeric target has few unique values, so it was treated as classification.")
    else:
        task = "regression"
    return task, warnings


def _get_estimator(name: str, task: str, class_weight=None):
    cw = "balanced" if class_weight == "class_weight" else None
    if task == "classification":
        if name == "Logistic Regression":
            return LogisticRegression(class_weight=cw, max_iter=500)
        elif name == "Decision Tree Classifier":
            return DecisionTreeClassifier(class_weight=cw)
        elif name == "Random Forest Classifier":
            return RandomForestClassifier(class_weight=cw, n_jobs=TRAIN_JOBS)
        elif name == "Support Vector Machine":
            return SVC(class_weight=cw)
        elif name == "K-Nearest Neighbors":
            return KNeighborsClassifier()
        elif name == "Naive Bayes":
            return GaussianNB()
        elif name == "Gradient Boosting Classifier":
            return GradientBoostingClassifier()
        elif name == "Extra Trees Classifier":
            return ExtraTreesClassifier(class_weight=cw, n_jobs=TRAIN_JOBS)
    else:
        if name == "Linear Regression":
            return LinearRegression()
        elif name == "Ridge Regression":
            return Ridge()
        elif name == "Decision Tree Regressor":
            return DecisionTreeRegressor()
        elif name == "Random Forest Regressor":
            return RandomForestRegressor(n_jobs=TRAIN_JOBS)
        elif name == "Gradient Boosting Regressor":
            return GradientBoostingRegressor()
        elif name == "Extra Trees Regressor":
            return ExtraTreesRegressor(n_jobs=TRAIN_JOBS)
        elif name == "Support Vector Regressor":
            return SVR()
    raise ValueError(f"Unknown model: {name}")


def _build_preprocessor(X: pd.DataFrame, scale: bool, dense_output: bool = False):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))

    num_pipe = Pipeline(num_steps)
    try:
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=2,
            max_categories=64,
            sparse_output=not dense_output,
        )
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=not dense_output)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0 if dense_output else 1.0,
    )


def train_models(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    model_names: List[str],
    imbalance_strategy: Optional[str],
    cv_folds: int = 5,
    test_size: float = 0.2,
) -> List[Dict[str, Any]]:
    X = df.drop(columns=[target])
    y = df[target].copy()

    # Encode target if classification
    le = None
    if task_type == "classification" and y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)

    stratify = None
    min_class_count = None
    if task_type == "classification":
        class_counts = y.value_counts()
        min_class_count = int(class_counts.min()) if len(class_counts) else 0
        if len(class_counts) < 2:
            raise ValueError("Classification target must contain at least two classes.")
        if min_class_count >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=stratify,
    )

    smote_step = None
    if imbalance_strategy == "smote" and task_type == "classification":
        try:
            from imblearn.over_sampling import SMOTE
            min_train_class = int(y_train.value_counts().min())
            if min_train_class >= 2:
                smote_step = SMOTE(random_state=42, k_neighbors=min(5, min_train_class - 1))
            else:
                imbalance_strategy = "class_weight"
        except ImportError:
            imbalance_strategy = "class_weight"

    grids = CLASSIFICATION_GRIDS if task_type == "classification" else REGRESSION_GRIDS
    if task_type == "classification":
        train_min_class = int(y_train.value_counts().min())
        effective_cv = max(2, min(cv_folds, MAX_CV_FOLDS, train_min_class))
        cv = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=42)
        scoring = "f1_weighted"
    else:
        effective_cv = max(2, min(cv_folds, MAX_CV_FOLDS, len(X_train)))
        cv = KFold(n_splits=effective_cv, shuffle=True, random_state=42)
        scoring = "neg_root_mean_squared_error"

    results = []
    for model_name in model_names:
        estimator = preprocessor = pipeline = search = best_estimator = None
        try:
            use_cw = (
                imbalance_strategy == "class_weight"
                and model_name in SUPPORTS_CLASS_WEIGHT
            )
            estimator = _get_estimator(
                model_name, task_type,
                class_weight="class_weight" if use_cw else None,
            )
            scale = model_name in NEEDS_SCALING
            preprocessor = _build_preprocessor(
                X_train,
                scale,
                dense_output=model_name in DENSE_PREPROCESSING_MODELS,
            )

            if smote_step is not None:
                from imblearn.pipeline import Pipeline as ImbPipeline
                pipeline = ImbPipeline([
                    ("preprocessor", preprocessor),
                    ("smote", smote_step),
                    ("clf", estimator),
                ])
            else:
                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("clf", estimator),
                ])

            param_grid = grids.get(model_name)
            if not param_grid:
                raise ValueError(f"No hyperparameter grid configured for {model_name}.")
            search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=TRAIN_JOBS,
                pre_dispatch=TRAIN_JOBS,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
            best_params = {
                k.replace("clf__", ""): v
                for k, v in search.best_params_.items()
            }
            cv_score = abs(search.best_score_)

            y_pred = best_estimator.predict(X_test)
            result: Dict[str, Any] = {
                "model_name": model_name,
                "best_params": best_params,
                "cv_score": round(float(cv_score), 4),
                "cv_folds": effective_cv,
                "training_rows": int(len(X_train)),
                "used_smote": imbalance_strategy == "smote",
                "used_class_weight": use_cw,
            }

            if task_type == "classification":
                result["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
                result["precision"] = round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                result["recall"] = round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                result["f1"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                result["primary_score"] = result["f1"]

                try:
                    if hasattr(best_estimator, "predict_proba"):
                        y_prob = best_estimator.predict_proba(X_test)
                        if y_prob.shape[1] == 2:
                            result["roc_auc"] = round(float(roc_auc_score(y_test, y_prob[:, 1])), 4)
                        else:
                            result["roc_auc"] = round(float(roc_auc_score(y_test, y_prob, multi_class="ovr")), 4)
                except Exception:
                    result["roc_auc"] = None

                result["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
            else:
                result["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
                result["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
                result["r2"] = round(float(r2_score(y_test, y_pred)), 4)
                result["primary_score"] = -result["rmse"]

            # AI explanation for best model (generated after comparison)
            result["ai_explanation"] = None
            results.append(result)

        except Exception as e:
            results.append({
                "model_name": model_name,
                "error": str(e),
                "primary_score": -999,
            })
        finally:
            # Grid searches retain fitted candidates until their references are released.
            # Collect between models so sequential searches remain inside hosted RAM limits.
            best_estimator = search = pipeline = preprocessor = estimator = None
            gc.collect()

    # Select best and generate explanation
    valid = [r for r in results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r.get("primary_score", -999))
        best["reason"] = _build_reason(best, task_type)
        try:
            best["ai_explanation"] = generate_explanation(
                f"Explain why {best['model_name']} was chosen as the best model for this "
                f"{'classification' if task_type == 'classification' else 'regression'} task. "
                f"Key metrics: {best}. Keep it beginner-friendly."
            )
        except Exception:
            best["ai_explanation"] = None

    return results


def _build_reason(result: Dict, task_type: str) -> str:
    name = result["model_name"]
    if task_type == "classification":
        return (
            f"{name} achieved the highest weighted F1-score ({result.get('f1', 0):.3f}), "
            f"with accuracy of {result.get('accuracy', 0):.3f}. "
            f"It was selected because it balanced precision and recall best among all trained models."
        )
    else:
        return (
            f"{name} achieved the lowest RMSE ({result.get('rmse', 0):.4f}) "
            f"and R2 of {result.get('r2', 0):.3f}, "
            f"making it the most accurate model for this regression task."
        )


def train_all_models(
    df: pd.DataFrame,
    target_col: str,
    selected_models: List[str],
    task_type: str,
    imbalance_strategy: Optional[str] = None,
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper for the FastAPI router layer."""
    strategy = None if imbalance_strategy == "none" else imbalance_strategy
    results = train_models(
        df=df,
        target=target_col,
        task_type=task_type,
        model_names=selected_models,
        imbalance_strategy=strategy,
        cv_folds=cv_folds,
        test_size=test_size,
    )
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {
            "results": [],
            "best_model": "",
            "best_model_params": {},
            "best_score": 0,
            "selection_reason": "All selected models failed.",
            "imbalance_strategy": imbalance_strategy or "none",
        }

    best = max(valid, key=lambda r: r.get("primary_score", -999))
    api_results = []
    for r in valid:
        converted = {
            "model_name": r["model_name"],
            "task_type": task_type,
            "best_cv_score": r.get("cv_score", 0),
            "best_params": r.get("best_params", {}),
            "smote_applied": r.get("used_smote", False),
            "class_weight_applied": r.get("used_class_weight", False),
            "is_best": r["model_name"] == best["model_name"],
            "confusion_matrix": r.get("confusion_matrix"),
        }
        if task_type == "classification":
            converted.update({
                "test_accuracy": r.get("accuracy"),
                "test_precision": r.get("precision"),
                "test_recall": r.get("recall"),
                "test_f1": r.get("f1"),
                "test_roc_auc": r.get("roc_auc"),
            })
        else:
            converted.update({
                "test_mae": r.get("mae"),
                "test_rmse": r.get("rmse"),
                "test_r2": r.get("r2"),
            })
        api_results.append(converted)

    return {
        "results": api_results,
        "best_model": best["model_name"],
        "best_model_params": best.get("best_params", {}),
        "best_score": best.get("primary_score", 0),
        "selection_reason": best.get("reason", ""),
        "imbalance_strategy": imbalance_strategy or "none",
    }
