"""Stage: evaluate
Out-of-fold predictions for the final model's configuration (per-subtype metrics,
confusion matrix) and a drop-one-stain analysis of the IHC panel.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate)
from sklearn.pipeline import Pipeline

from common import (METRICS, PLOTS, ihc_columns, load_dataset, load_params, make_model,
                    make_pipeline, preprocessor)


def main():
    params = load_params()
    subtypes = params["data"]["subtypes"]
    seed = params["cv"]["random_state"]
    data = load_dataset(params)
    y = data["Subtype"].to_numpy(dtype=object)
    fm = params["final_model"]
    PLOTS.mkdir(exist_ok=True)
    METRICS.mkdir(exist_ok=True)

    # Out-of-fold predictions: each patient predicted by a model that never saw them
    pipe, cols = make_pipeline(fm["feature_set"], fm["model"], params)
    cv5 = StratifiedKFold(n_splits=params["cv"]["n_splits"], shuffle=True, random_state=seed)
    pred = cross_val_predict(pipe, data[cols], y, cv=cv5)

    report = classification_report(y, pred, labels=subtypes, output_dict=True, zero_division=0)
    per_class = {s: {k: round(report[s][k], 4) for k in ("precision", "recall", "f1-score")} for s in subtypes}
    (METRICS / "per_class.json").write_text(json.dumps(per_class, indent=2))
    pd.DataFrame({"actual": y, "predicted": pred}, index=data.index).to_csv(METRICS / "oof_predictions.csv")

    cm = confusion_matrix(y, pred, labels=subtypes, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4.3))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False,
                xticklabels=subtypes, yticklabels=subtypes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{fm['model']} on {fm['feature_set']}")
    plt.tight_layout(); plt.savefig(PLOTS / "confusion_matrix.png", dpi=150); plt.close()

    # Drop-one-stain analysis on the IHC panel
    stains = {"AR": ["IHC_AR"], "Ki-67": ["IHC_MKI67"], "CK5/6": ["IHC_KRT5", "IHC_KRT6A"],
              "EGFR": ["IHC_EGFR"], "Vimentin": ["IHC_VIM"], "FOXA1": ["IHC_FOXA1"]}
    ihc = ihc_columns(params)
    cv_drop = RepeatedStratifiedKFold(n_splits=params["cv"]["n_splits"],
                                      n_repeats=params["evaluate"]["ablation_repeats"], random_state=seed)

    def score(c):
        p = Pipeline([("prep", preprocessor(c, [])), ("model", make_model("random_forest", params))])
        s = cross_validate(p, data[c], y, cv=cv_drop, scoring="f1_macro")["test_score"]
        return s.mean(), s.std() / len(s) ** 0.5   # mean and approx. standard error (folds overlap, so treat as rough)

    full, _ = score(ihc)
    rows = []
    for s_name, cs in stains.items():
        m, se = score([c for c in ihc if c not in cs])
        rows.append({"stain_removed": s_name, "f1_macro_without": round(m, 4), "std_error": round(se, 4)})

    drop = pd.DataFrame(rows)
    drop["f1_macro_drop"] = (full - drop["f1_macro_without"]).round(4)
    drop.sort_values("f1_macro_drop", ascending=False).to_csv(METRICS / "stain_ablation.csv", index=False)

    d = drop.sort_values("f1_macro_drop")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(d["stain_removed"], d["f1_macro_drop"], xerr=d["std_error"], color="#2f6fae", capsize=3)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Drop in macro F1 when the stain is removed"); ax.set_ylabel("")
    ax.set_title(f"Stain importance (full panel macro F1 = {full:.3f})")
    plt.tight_layout(); plt.savefig(PLOTS / "stain_ablation.png", dpi=150); plt.close()

    print(json.dumps(per_class, indent=2))
    print(drop.to_string(index=False))


if __name__ == "__main__":
    main()
