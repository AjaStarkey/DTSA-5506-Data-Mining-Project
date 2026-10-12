"""Stage: train
Repeated stratified k-fold CV for every (feature set x model) combination,
then refit the chosen final model on all patients and save it.
"""
import json

import joblib
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from common import METRICS, MODELS_DIR, feature_sets, load_dataset, load_params, make_pipeline


def main():
    params = load_params()
    data = load_dataset(params)
    y = data["Subtype"].to_numpy(dtype=object)
    cvp = params["cv"]
    cv = RepeatedStratifiedKFold(n_splits=cvp["n_splits"], n_repeats=cvp["n_repeats"],
                                 random_state=cvp["random_state"])

    rows, metrics = [], {}
    for fs in feature_sets(params):
        for model_name in params["models"]:
            pipe, cols = make_pipeline(fs, model_name, params)
            s = cross_validate(pipe, data[cols], y, cv=cv, scoring=["f1_macro", "accuracy"])
            res = {"f1_macro_mean": round(float(s["test_f1_macro"].mean()), 4),
                   "f1_macro_sd": round(float(s["test_f1_macro"].std()), 4),
                   "accuracy_mean": round(float(s["test_accuracy"].mean()), 4)}
            metrics.setdefault(fs, {})[model_name] = res
            rows.append({"feature_set": fs, "model": model_name, **res})
            print(f"{fs:14s} {model_name:20s} macro F1 = {res['f1_macro_mean']:.3f} ± {res['f1_macro_sd']:.3f}")

    METRICS.mkdir(exist_ok=True)
    (METRICS / "cv_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(rows).to_csv(METRICS / "cv_results.csv", index=False)

    # Refit the chosen model on all patients for serving
    fm = params["final_model"]
    pipe, cols = make_pipeline(fm["feature_set"], fm["model"], params)
    pipe.fit(data[cols], y)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": cols, "classes": list(pipe.classes_)},
                MODELS_DIR / "model.joblib")
    print(f"Saved final model: {fm['model']} on {fm['feature_set']} ({len(cols)} features)")


if __name__ == "__main__":
    main()
