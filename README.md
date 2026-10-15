# TNBC Subtype Classification: Routine Clinical Data vs. a Low-Cost IHC Panel

Can triple-negative breast cancer (TNBC) be subtyped without tumor sequencing? This project compares routine clinical data with a six-stain immunohistochemistry (IHC) panel for predicting the four Lehmann TNBC subtypes (BL1, BL2, LAR, M) in 175 TCGA patients. It is also a hands-on MLOps project: data and pipeline versioning with **DVC**, experiment tracking and a model registry with **MLflow**, and orchestration with **Dagster**, with KServe serving planned next.

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

## Dagster: orchestration, quality checks and model promotion

`orchestration/definitions.py` turns the pipeline into five Dagster assets:

```
raw_data -> processed_dataset -> trained_models -> model_evaluation -> champion_model
```

- Each asset runs its DVC stage (`dvc repro <stage>`), so DVC still decides what needs rebuilding and skips anything unchanged.
- **Asset checks** run after each step: all four subtypes and 175 patients present, no missing IHC values, final model macro F1 >= 0.50 (a blocking quality gate), IHC panel beats routine clinical data, and every subtype has F1 >= 0.35.
- **champion_model** gives the newest registered MLflow model version the alias `champion` if it passes the quality gate. Serving (KServe) will load `models:/tnbc-subtype-classifier@champion`.
- **Automation:** a weekly retraining schedule (Mondays 6 am Mountain) and a sensor that starts the pipeline when `data/raw.dvc` or `params.yaml` changes. Both start switched off; turn them on in the UI under **Automation**.

```bash
bash dagster_ui.sh      # open port 3000, then Assets -> Materialize all
dagster job execute -f orchestration/definitions.py -j full_pipeline   # or run it from the terminal
```

## Data sources
- **Clinical data and RNA-seq:** TCGA-BRCA PanCancer Atlas (Liu et al., 2018, *Cell*), via the cBioPortal datahub
- **Subtype labels:** Lehmann et al. (2016), *PLoS ONE* 11(6): e0157368, Supplementary Table S1 ("Refined TNBCtype")

## Roadmap
- [x] **DVC:** data versioning, reproducible pipeline, metrics and experiments
- [x] **MLflow:** experiment tracking and model registry
- [x] **Dagster:** orchestrate the pipeline as scheduled, monitored assets with quality checks and model promotion
- [ ] **KServe:** serve the final model behind a prediction API
