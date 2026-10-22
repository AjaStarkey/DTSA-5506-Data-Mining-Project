"""Re-run the cross-validation for all feature sets and models and check that the results
match the committed metrics/cv_results.csv. Nothing is logged to MLflow or overwritten.

    python src/verify_reproducible.py
"""
import sys

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from common import METRICS, feature_sets, load_dataset, load_params, make_pipeline

TOL = 0.0005  # the CSV stores values rounded to 4 decimals

params = load_params()
data = load_dataset(params)
y = data["Subtype"].to_numpy(dtype=object)
cvp = params["cv"]
cv = RepeatedStratifiedKFold(n_splits=cvp["n_splits"], n_repeats=cvp["n_repeats"],
                             random_state=cvp["random_state"])
saved = pd.read_csv(METRICS / "cv_results.csv").set_index(["feature_set", "model"])

failures = 0
print(f"{'feature set':14s} {'model':20s} {'saved':>7s} {'rerun':>7s}")
for fs in feature_sets(params):
    for model_name in params["models"]:
        pipe, cols = make_pipeline(fs, model_name, params)
        s = cross_validate(pipe, data[cols], y, cv=cv, scoring=["f1_macro"])
        new = float(s["test_f1_macro"].mean())
        old = float(saved.loc[(fs, model_name), "f1_macro_mean"])
        ok = abs(new - old) <= TOL
        failures += not ok
        print(f"{fs:14s} {model_name:20s} {old:7.4f} {new:7.4f}  {'match' if ok else 'DIFFERENT'}")

if failures:
    sys.exit(f"\n{failures} result(s) differ from metrics/cv_results.csv. The installed package "
             "versions probably differ from requirements.lock.txt.")
print("\nAll results match metrics/cv_results.csv: this environment reproduces the reported numbers.")
