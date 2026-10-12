#!/usr/bin/env bash
# One-step update: adds the original 12-feature version (clinical_original),
# re-versions the data with DVC, reruns the pipeline, commits and pushes.
set -e
cd "$(dirname "$0")"
grep -qx "run_update.sh" .git/info/exclude 2>/dev/null || echo "run_update.sh" >> .git/info/exclude   # keep this script out of git

echo "== 1/6  Updating project files =="
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
EOF_TNBC
cat > src/common.py <<'EOF_TNBC'
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


def categorical_columns(params):
    f = params["features"]
    return f["clinical_categorical"] + f["original_extra_categorical"]


def feature_sets(params):
    """Name -> (numeric columns, categorical columns)."""
    f = params["features"]
    num, cat = f["clinical_numeric"], f["clinical_categorical"]
    ihc = ihc_columns(params)
    return {
        # Original course version: 12 clinical features incl. Buffa hypoxia (mRNA-derived)
        "clinical_original": (num + f["original_extra_numeric"], cat + f["original_extra_categorical"]),
        # Routine clinical features only (9)
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
    cat = categorical_columns(params)
    data[cat] = data[cat].astype(object).where(data[cat].notna(), np.nan)
    data["Subtype"] = data["Subtype"].astype(str).to_numpy(dtype=object)
    return data
EOF_TNBC
cat > src/download_data.py <<'EOF_TNBC'
"""One-time download of the raw data into data/raw/.

After running this, version the raw data with:  dvc add data/raw
"""
import urllib.request

import requests

from common import RAW

DATAHUB = ("https://media.githubusercontent.com/media/cBioPortal/datahub/master/"
           "public/brca_tcga_pan_can_atlas_2018/")
FILES = {
    "tcga_brca_clinical_patient.txt": DATAHUB + "data_clinical_patient.txt",
    "tcga_brca_mrna_rsem.txt": DATAHUB + "data_mrna_seq_v2_rsem.txt",  # ~150 MB
    "tcga_brca_hypoxia.txt": DATAHUB + "data_clinical_supp_hypoxia.txt",  # Buffa score (original version only)
}
S1_URL = ("https://journals.plos.org/plosone/article/file?type=supplementary"
          "&id=10.1371/journal.pone.0157368.s006")
S1_FILE = "lehmann_2016_S1.xlsx"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW / name
        if dest.exists():
            print(f"✓ {name} already present")
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(url, dest)

    dest = RAW / S1_FILE
    if dest.exists():
        print(f"✓ {S1_FILE} already present")
    else:
        try:
            r = requests.get(S1_URL, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"Downloaded {S1_FILE}")
        except Exception as e:  # PLOS sometimes blocks scripted downloads
            print(f"Could not download the Lehmann S1 table ({e}).\n"
                  "Download it manually from https://doi.org/10.1371/journal.pone.0157368 "
                  f"(Supporting Information, S1 Table) and save it as data/raw/{S1_FILE}")
    print("Raw data ready in data/raw/")


if __name__ == "__main__":
    main()
EOF_TNBC
cat > src/prepare.py <<'EOF_TNBC'
"""Stage: prepare
Raw files -> one modeling table (data/processed/dataset.parquet):
Lehmann subtype labels + clinical features (9 routine + 3 used only by the original
12-feature version) + 7 IHC stand-in genes.
"""
import json

import numpy as np
import pandas as pd

from common import PROCESSED, RAW, ihc_columns, load_params


def load_labels(subtypes):
    s1 = pd.read_excel(RAW / "lehmann_2016_S1.xlsx").iloc[1:]  # first row is header text
    labels = pd.DataFrame({
        "TCGA_ID": s1.iloc[:, 0].astype(str).str[:12],          # barcode -> patient ID
        "Subtype": s1.iloc[:, 5].astype(str).str.strip(),       # "Refined TNBCtype"
    })
    labels = labels[labels["TCGA_ID"].str.startswith("TCGA") & labels["Subtype"].isin(subtypes)]
    return labels.drop_duplicates("TCGA_ID").set_index("TCGA_ID")


def load_clinical(params):
    f = params["features"]
    num = f["clinical_numeric"] + f["original_extra_numeric"]
    cat = f["clinical_categorical"] + f["original_extra_categorical"]
    clin = pd.read_csv(RAW / "tcga_brca_clinical_patient.txt", sep="\t", comment="#").set_index("PATIENT_ID")
    hyp = pd.read_csv(RAW / "tcga_brca_hypoxia.txt", sep="\t", comment="#").set_index("PATIENT_ID")
    clin = clin.join(hyp[["BUFFA_HYPOXIA_SCORE"]])
    clin = clin[num + cat].copy()
    clin[num] = clin[num].apply(pd.to_numeric, errors="coerce")
    clin[cat] = clin[cat].astype("string")
    return clin


def load_ihc(genes):
    expr = pd.read_csv(RAW / "tcga_brca_mrna_rsem.txt", sep="\t")
    expr = expr[expr["Hugo_Symbol"].isin(genes)].drop(columns="Entrez_Gene_Id")
    expr = expr.groupby("Hugo_Symbol").mean()
    expr = expr.loc[:, expr.columns.str.endswith("-01")]       # primary tumors
    expr.columns = expr.columns.str[:12]
    expr = expr.loc[:, ~expr.columns.duplicated()]
    ihc = np.log2(expr.clip(lower=0) + 1).T                    # log2(RSEM + 1)
    ihc.columns = [f"IHC_{g}" for g in ihc.columns]
    return ihc


def main():
    params = load_params()
    labels = load_labels(params["data"]["subtypes"])
    clinical = load_clinical(params)
    ihc = load_ihc(list(params["features"]["ihc_genes"]))

    data = labels.join(clinical, how="inner").join(ihc, how="inner")
    data = data[[c for c in clinical.columns] + ihc_columns(params) + ["Subtype"]]
    data.index.name = "TCGA_ID"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    data.to_parquet(PROCESSED / "dataset.parquet")

    summary = {
        "n_patients": int(len(data)),
        "n_labeled": int(len(labels)),
        "subtype_counts": data["Subtype"].value_counts().to_dict(),
        "age_median": float(data["AGE"].median()),
    }
    (PROCESSED / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
EOF_TNBC
cat > README.md <<'EOF_TNBC'
# TNBC Subtype Classification: Routine Clinical Data vs. a Low-Cost IHC Panel

Can triple-negative breast cancer (TNBC) be subtyped without tumor sequencing? This project compares routine clinical data with a six-stain immunohistochemistry (IHC) panel for predicting the four Lehmann TNBC subtypes (BL1, BL2, LAR, M) in 175 TCGA patients. It is also a hands-on MLOps project: data and pipeline versioning with **DVC**, with MLflow, Dagster and KServe planned next.

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

## Data sources
- **Clinical data and RNA-seq:** TCGA-BRCA PanCancer Atlas (Liu et al., 2018, *Cell*), via the cBioPortal datahub
- **Subtype labels:** Lehmann et al. (2016), *PLoS ONE* 11(6): e0157368, Supplementary Table S1 ("Refined TNBCtype")

## Roadmap
- [x] **DVC:** data versioning, reproducible pipeline, metrics and experiments
- [ ] **MLflow:** experiment tracking and model registry
- [ ] **Dagster:** orchestrate the pipeline as scheduled, monitored assets
- [ ] **KServe:** serve the final model behind a prediction API
EOF_TNBC

if [ -f update_tnbc.sh ]; then git rm -q update_tnbc.sh 2>/dev/null || rm -f update_tnbc.sh; fi
grep -q clinical_original params.yaml && echo "Project files updated"

echo "== 2/6  Downloading new raw data =="
python src/download_data.py

echo "== 3/6  Versioning data with DVC =="
dvc add data/raw
git add -A
git commit -q -m "Add original 12-feature version (clinical_original)" || true

echo "== 4/6  Running the pipeline (5-8 minutes) =="
dvc repro

echo "== 5/6  Saving results =="
git add -A
git commit -q -m "Rerun pipeline with original 12-feature version" || true

echo "== 6/6  Pushing to GitHub =="
git push || echo "Push did not work - send this message to Claude"

echo
dvc metrics show
echo
echo "ALL DONE"
