"""Stage: train
Repeated stratified k-fold CV for every (feature set x model) combination,
then refit the chosen final model on all patients and save it.

Every run is also logged to MLflow:
  - one parent run per `dvc repro`, holding all settings and the DVC data fingerprint
  - one child run per (feature set x model) combination with its CV scores
  - the final model, registered in the MLflow model registry
"""
import inspect
import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from common import METRICS, MODELS_DIR, ROOT, feature_sets, load_dataset, load_params, make_pipeline


def flatten(d, prefix=""):
    """{'cv': {'n_splits': 5}} -> {'cv.n_splits': 5}"""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v if not isinstance(v, list) else ",".join(map(str, v))
    return out


def dvc_data_md5():
    """Fingerprint of data/raw as recorded by DVC (links the MLflow run to the exact dataset)."""
    try:
        with open(ROOT / "data" / "raw.dvc") as f:
            return yaml.safe_load(f)["outs"][0]["md5"]
    except Exception:
        return "unknown"


def main():
    params = load_params()
    data = load_dataset(params)
    y = data["Subtype"].to_numpy(dtype=object)
    cvp = params["cv"]
    cv = RepeatedStratifiedKFold(n_splits=cvp["n_splits"], n_repeats=cvp["n_repeats"],
                                 random_state=cvp["random_state"])

    mlp = params["mlflow"]
    mlflow.set_tracking_uri(mlp["tracking_uri"])
    mlflow.set_experiment(mlp["experiment_name"])

    rows, metrics = [], {}
    with mlflow.start_run(run_name="dvc-train") as parent:
        mlflow.set_tags({"dvc_data_md5": dvc_data_md5(), "stage": "train"})
        mlflow.log_params({k: v for k, v in flatten(params).items()
                           if k.split(".")[0] in ("cv", "models", "final_model")})
        mlflow.log_param("n_patients", len(data))

        for fs in feature_sets(params):
            for model_name in params["models"]:
                pipe, cols = make_pipeline(fs, model_name, params)
                s = cross_validate(pipe, data[cols], y, cv=cv, scoring=["f1_macro", "accuracy"])
                res = {"f1_macro_mean": round(float(s["test_f1_macro"].mean()), 4),
                       "f1_macro_sd": round(float(s["test_f1_macro"].std()), 4),
                       "accuracy_mean": round(float(s["test_accuracy"].mean()), 4)}
                metrics.setdefault(fs, {})[model_name] = res
                rows.append({"feature_set": fs, "model": model_name, **res})
                print(f"{fs:17s} {model_name:20s} macro F1 = {res['f1_macro_mean']:.3f} ± {res['f1_macro_sd']:.3f}")

                with mlflow.start_run(run_name=f"{fs}__{model_name}", nested=True):
                    mlflow.log_params({"feature_set": fs, "model": model_name, "n_features": len(cols),
                                       **{f"hp.{k}": v for k, v in params["models"][model_name].items()}})
                    mlflow.log_metrics(res)

        METRICS.mkdir(exist_ok=True)
        (METRICS / "cv_metrics.json").write_text(json.dumps(metrics, indent=2))
        pd.DataFrame(rows).to_csv(METRICS / "cv_results.csv", index=False)
        mlflow.log_artifact(str(METRICS / "cv_results.csv"))

        # Refit the chosen model on all patients for serving
        fm = params["final_model"]
        pipe, cols = make_pipeline(fm["feature_set"], fm["model"], params)
        pipe.fit(data[cols], y)
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(pipe, MODELS_DIR / "model.joblib")               # plain scikit-learn pipeline
        (MODELS_DIR / "model_features.json").write_text(json.dumps(
            {"feature_set": fm["feature_set"], "model": fm["model"], "features": cols,
             "classes": [str(c) for c in pipe.classes_]}, indent=2))

        final = metrics[fm["feature_set"]][fm["model"]]
        mlflow.log_metrics({f"final.{k}": v for k, v in final.items()})
        log_kwargs = dict(name="model", input_example=data[cols].head(3),
                          registered_model_name=mlp["registered_model_name"])
        # Newer MLflow saves scikit-learn models with skops (a safer format than pickle) and
        # only reloads object types it is told to trust. These are the types a random forest uses.
        if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
            log_kwargs["skops_trusted_types"] = ["numpy.dtype", "sklearn.tree._tree.Tree"]
        mlflow.sklearn.log_model(pipe, **log_kwargs)
        print(f"Saved final model: {fm['model']} on {fm['feature_set']} ({len(cols)} features)")
        print(f"MLflow run: {parent.info.run_id}  (view with: mlflow ui --backend-store-uri {mlp['tracking_uri']})")


if __name__ == "__main__":
    main()
