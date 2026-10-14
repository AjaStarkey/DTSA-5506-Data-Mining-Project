#!/usr/bin/env bash
# Adds MLflow experiment tracking + model registry to the TNBC project,
# reruns the pipeline (logging to MLflow), commits and pushes.
set -e
cd "$(dirname "$0")"
grep -qx "add_mlflow.sh" .git/info/exclude 2>/dev/null || echo "add_mlflow.sh" >> .git/info/exclude   # keep this script out of git

echo "== 1/5  Updating project files =="
cat > params.yaml <<'EOF_TNBC'
# All tunable settings live here. DVC tracks this file, so changing a value
# marks the affected pipeline stages as out of date.

data:
  subtypes: [BL1, BL2, LAR, M]

features:
  clinical_numeric: [AGE]
  clinical_categorical:
    - AJCC_PATHOLOGIC_TUMOR_STAGE
    - PATH_T_STAGE
    - PATH_N_STAGE
    - PATH_M_STAGE
    - ICD_O_3_HISTOLOGY
    - RACE
    - ETHNICITY
    - PRIOR_DX
  # Extra features used only by the ORIGINAL 12-feature course version (clinical_original).
  # Excluded from the routine model: Buffa is calculated from tumor mRNA, staging edition
  # reflects diagnosis year/hospital, and radiation therapy is given after diagnosis.
  original_extra_numeric: [BUFFA_HYPOXIA_SCORE]
  original_extra_categorical: [AJCC_STAGING_EDITION, RADIATION_THERAPY]
  # Genes used as mRNA stand-ins for IHC stains
  ihc_genes:
    AR: Androgen receptor
    FOXA1: FOXA1
    MKI67: Ki-67
    KRT5: CK5/6
    KRT6A: CK5/6
    EGFR: EGFR
    VIM: Vimentin

cv:
  n_splits: 5
  n_repeats: 5
  random_state: 42

models:
  logistic_regression:
    C: 0.1
  random_forest:
    n_estimators: 300
    min_samples_leaf: 3

evaluate:
  ablation_repeats: 10          # repeats for the drop-one-stain analysis (more = less noise)

# Model refit on all patients and saved for serving
final_model:
  feature_set: ihc_panel        # clinical_original | clinical | ihc_panel | clinical_ihc
  model: random_forest          # logistic_regression | random_forest

# Experiment tracking. A local SQLite file in the project folder; ignored by git.
mlflow:
  tracking_uri: sqlite:///mlflow.db
  experiment_name: tnbc-subtype-classification
  registered_model_name: tnbc-subtype-classifier
EOF_TNBC
cat > dvc.yaml <<'EOF_TNBC'
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - src/prepare.py
      - src/common.py
      - data/raw
    params:
      - data
      - features
    outs:
      - data/processed/dataset.parquet
    metrics:
      - data/processed/data_summary.json:
          cache: false

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - src/common.py
      - data/processed/dataset.parquet
    params:
      - features
      - cv
      - models
      - final_model
      - mlflow
    outs:
      - models/model.joblib
      - models/model_features.json:
          cache: false
    metrics:
      - metrics/cv_metrics.json:
          cache: false
    plots:
      - metrics/cv_results.csv:
          cache: false
          template: bar_horizontal
          x: f1_macro_mean
          y: feature_set

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - src/common.py
      - data/processed/dataset.parquet
    params:
      - cv
      - models
      - final_model
      - evaluate
    metrics:
      - metrics/per_class.json:
          cache: false
    outs:
      - metrics/stain_ablation.csv:
          cache: false
    plots:
      - metrics/oof_predictions.csv:
          cache: false
          template: confusion
          x: actual
          y: predicted
      - plots/confusion_matrix.png:
          cache: false
      - plots/stain_ablation.png:
          cache: false
EOF_TNBC
cat > requirements.txt <<'EOF_TNBC'
dvc>=3.50
pandas>=2.0
numpy
scikit-learn>=1.4
matplotlib
seaborn
openpyxl
pyarrow
pyyaml
requests
joblib
mlflow>=3.0
EOF_TNBC
cat > .gitignore <<'EOF_TNBC'
__pycache__/
*.pyc
.venv/
.ipynb_checkpoints/
dvc_plots/
mlflow.db
mlruns/
mlartifacts/
.DS_Store
EOF_TNBC
cat > src/train.py <<'EOF_TNBC'
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
EOF_TNBC
cat > README.md <<'EOF_TNBC'
# TNBC Subtype Classification: Routine Clinical Data vs. a Low-Cost IHC Panel

Can triple-negative breast cancer (TNBC) be subtyped without tumor sequencing? This project compares routine clinical data with a six-stain immunohistochemistry (IHC) panel for predicting the four Lehmann TNBC subtypes (BL1, BL2, LAR, M) in 175 TCGA patients. It is also a hands-on MLOps project: data and pipeline versioning with **DVC**, experiment tracking and a model registry with **MLflow**, with Dagster and KServe planned next.

## Results (repeated stratified 5-fold CV, 25 fits per model)

| Features | Best macro F1 |
|---|---|
| Original course version (12 clinical features, incl. Buffa hypoxia score) | ~0.34 |
| Routine clinical only (9 features) | ~0.27, about chance (0.25) |
| IHC panel (6 stains, mRNA stand-ins) | ~0.60 |
| Clinical + IHC panel | ~0.60 |

- **Original version:** my course project reached macro F1 0.36 on a single test split (~0.34 here with repeated CV). Most of that signal came from the Buffa hypoxia score, which is calculated from tumor mRNA, so it is not a routine clinical measurement.
- Routine clinical data alone cannot identify subtypes. The IHC panel roughly doubles macro F1.
- LAR and BL1 are identified best. BL2 and M remain hard to separate.
- Vimentin is the one stain whose removal clearly hurts performance (macro F1 drops ~0.06). Removing any other stain changes it by 0.02 or less, because the stains overlap (e.g., AR and FOXA1 both mark LAR).
- **Limitation:** TCGA has no results for these stains, so each stain is represented by the tumor's mRNA level for the matching gene. Real IHC is scored more coarsely, so real-world performance would likely be lower.

## Project structure

```
├── data/
│   ├── raw/                 # tracked by DVC (data/raw.dvc), not by git
│   └── processed/           # pipeline output (dataset.parquet)
├── src/
│   ├── download_data.py     # one-time raw data download
│   ├── prepare.py           # stage 1: labels + clinical + IHC genes -> parquet
│   ├── train.py             # stage 2: repeated CV for all feature sets/models, save final model
│   ├── evaluate.py          # stage 3: per-subtype metrics, confusion matrix, stain ablation
│   └── common.py            # shared feature lists and model builders
├── params.yaml              # every tunable setting
├── dvc.yaml                 # pipeline definition
├── dvc.lock                 # exact hashes of the last run (commit this)
├── metrics/  plots/  models/
└── notebooks/               # original exploratory Colab notebook
```

## Quick start

```bash
git clone https://github.com/AjaStarkey/tnbc-subtype-mlops.git
cd tnbc-subtype-mlops
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

dvc pull        # if a DVC remote is set up; otherwise run: python src/download_data.py
dvc repro       # runs prepare -> train -> evaluate (about 5 minutes)
dvc metrics show
```

## DVC walkthrough (first-time setup)

**1. Initialize git and DVC**
```bash
git init
dvc init
git commit -m "Initialize DVC"
```

**2. Download and version the raw data**
```bash
python src/download_data.py   # if the PLOS download fails, follow the printed instructions
dvc add data/raw
git add data/raw.dvc data/.gitignore
git commit -m "Track raw data with DVC"
```
`data/raw.dvc` is a small text file holding a hash of the data. Git versions that pointer, and DVC stores the actual ~150 MB of files.

**3. Run the pipeline**
```bash
dvc repro
git add dvc.lock metrics plots data/processed/.gitignore models/.gitignore
git commit -m "Run pipeline"
```
DVC runs the stages in order, using the dependencies listed in `dvc.yaml`. Run `dvc repro` again and nothing happens, because nothing has changed. Edit `src/evaluate.py` and only the `evaluate` stage reruns.

**4. Look at results**
```bash
dvc metrics show          # all tracked metrics
dvc plots show            # writes an HTML report of the plots
dvc dag                   # draws the pipeline graph
```

**5. Run experiments without editing files**
```bash
dvc exp run -n rf-500-trees -S models.random_forest.n_estimators=500
dvc exp run -n logreg-final -S final_model.model=logistic_regression
dvc exp show              # table comparing parameters and metrics across experiments
dvc exp apply rf-500-trees   # keep the one you like, then git commit
```

**6. Set up remote storage**
```bash
# Local folder (simplest, for practice)
dvc remote add -d localstore ../dvc-storage
# or S3 (needs: pip install "dvc[s3]" and AWS credentials)
# dvc remote add -d s3store s3://YOUR-BUCKET/tnbc-subtype-mlops
git add .dvc/config && git commit -m "Add DVC remote"
dvc push
```
Anyone who clones the repo can then run `dvc pull` to get the exact data used for every commit.

## MLflow: experiment tracking and model registry

Every `dvc repro` that reruns the train stage logs to MLflow (stored locally in `mlflow.db`, which git ignores):

- a parent run, **dvc-train**, with all settings from `params.yaml`, the git commit, and the DVC fingerprint of `data/raw` (tag `dvc_data_md5`), so each result can be traced to the exact code and data that produced it
- one child run per feature set and model, with its cross-validated macro F1 and accuracy
- the final model, registered as a new version of **tnbc-subtype-classifier**

Open the dashboard:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
In Codespaces, click **Open in Browser** when port 5000 is offered (or use the **Ports** tab).

## Data sources
- **Clinical data and RNA-seq:** TCGA-BRCA PanCancer Atlas (Liu et al., 2018, *Cell*), via the cBioPortal datahub
- **Subtype labels:** Lehmann et al. (2016), *PLoS ONE* 11(6): e0157368, Supplementary Table S1 ("Refined TNBCtype")

## Roadmap
- [x] **DVC:** data versioning, reproducible pipeline, metrics and experiments
- [x] **MLflow:** experiment tracking and model registry
- [ ] **Dagster:** orchestrate the pipeline as scheduled, monitored assets
- [ ] **KServe:** serve the final model behind a prediction API
EOF_TNBC
grep -q "mlflow" params.yaml && echo "Project files updated"

echo "== 2/5  Installing MLflow (1-2 minutes) =="
pip install -q -r requirements.txt
export MLFLOW_DISABLE_AGENT_HINT=1

echo "== 3/5  Rerunning the pipeline with MLflow logging (about 2 minutes) =="
dvc repro

echo "== 4/5  Saving to git =="
git add -A
git commit -q -m "Add MLflow experiment tracking and model registry" || true

echo "== 5/5  Pushing to GitHub =="
git push || echo "Push did not work - send this message to Claude"

echo
echo "ALL DONE. Open the MLflow dashboard with:"
echo "    mlflow ui --backend-store-uri sqlite:///mlflow.db"
