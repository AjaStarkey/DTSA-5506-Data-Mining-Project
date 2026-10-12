"""Shared helpers: parameter loading, feature sets and model builders."""
from pathlib import Path

import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
METRICS = ROOT / "metrics"
PLOTS = ROOT / "plots"
MODELS_DIR = ROOT / "models"


def load_params():
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def ihc_columns(params):
    return [f"IHC_{g}" for g in params["features"]["ihc_genes"]]


def feature_sets(params):
    """Name -> (numeric columns, categorical columns)."""
    num = params["features"]["clinical_numeric"]
    cat = params["features"]["clinical_categorical"]
    ihc = ihc_columns(params)
    return {
        "clinical": (num, cat),
        "ihc_panel": (ihc, []),
        "clinical_ihc": (num + ihc, cat),
    }


def preprocessor(num_cols, cat_cols):
    """Imputation, scaling and encoding. Fit inside each CV training fold, so no leakage."""
    steps = [("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), num_cols)]
    if cat_cols:
        steps.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols))
    return ColumnTransformer(steps)


def make_model(name, params):
    p = params["models"][name]
    seed = params["cv"]["random_state"]
    if name == "logistic_regression":
        return LogisticRegression(C=p["C"], class_weight="balanced", max_iter=5000)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=p["n_estimators"],
                                      min_samples_leaf=p["min_samples_leaf"],
                                      class_weight="balanced", random_state=seed, n_jobs=-1)
    raise ValueError(f"Unknown model: {name}")


def make_pipeline(feature_set, model_name, params):
    num, cat = feature_sets(params)[feature_set]
    return Pipeline([("prep", preprocessor(num, cat)), ("model", make_model(model_name, params))]), num + cat


def load_dataset(params):
    """Read the processed table and give categorical columns a plain object dtype
    (parquet can return Arrow-backed string dtypes, which scikit-learn handles poorly)."""
    import numpy as np
    import pandas as pd

    data = pd.read_parquet(PROCESSED / "dataset.parquet")
    cat = params["features"]["clinical_categorical"]
    data[cat] = data[cat].astype(object).where(data[cat].notna(), np.nan)
    data["Subtype"] = data["Subtype"].astype(str).to_numpy(dtype=object)
    return data
