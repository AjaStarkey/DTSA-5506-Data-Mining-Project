"""Dagster orchestration for the TNBC subtype pipeline.

DVC still owns *how* each stage is built (dvc.yaml) and caches results, so an
unchanged stage is skipped. Dagster adds what DVC doesn't: a visual asset graph,
run history and logs, data-quality checks, a schedule, a sensor that retrains when
data or settings change, and promotion of the best model in the MLflow registry.

Start the UI with:  bash dagster_ui.sh   (then open port 3000)
"""
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from dagster import (AssetCheckResult, AssetCheckSeverity, AssetExecutionContext,
                     AssetSelection, DefaultScheduleStatus, DefaultSensorStatus,
                     Definitions, MaterializeResult, MetadataValue, RunRequest,
                     ScheduleDefinition, SensorEvaluationContext, SkipReason,
                     asset, asset_check, define_asset_job, sensor)

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())
SUBTYPES = PARAMS["data"]["subtypes"]
F1_FLOOR = 0.50            # quality gate: the final model must reach this macro F1
EXPECTED_PATIENTS = 175


def dvc_repro(context, stage):
    """Run one DVC stage (and anything upstream that changed), streaming its log."""
    context.log.info(f"$ dvc repro {stage}")
    proc = subprocess.run(["dvc", "repro", stage], cwd=ROOT, capture_output=True, text=True)
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.strip():
            context.log.info(line)
    if proc.returncode != 0:
        raise RuntimeError(f"dvc repro {stage} failed (exit {proc.returncode})")
    skipped = "didn't change, skipping" in proc.stdout
    return proc.stdout, skipped


def md_table(df):
    return MetadataValue.md(df.to_markdown(index=False))


# ----------------------------------------------------------------------------- assets
@asset(group_name="tnbc_pipeline", description="Raw TCGA + Lehmann files, versioned by DVC (data/raw.dvc).")
def raw_data(context: AssetExecutionContext) -> MaterializeResult:
    raw = ROOT / "data" / "raw"
    files = sorted(p for p in raw.glob("*") if p.is_file())
    if not files:
        raise FileNotFoundError("data/raw is empty: run `dvc pull` or `python src/download_data.py`")
    info = yaml.safe_load((ROOT / "data" / "raw.dvc").read_text())["outs"][0]
    return MaterializeResult(metadata={
        "dvc_md5": info.get("md5", ""),
        "files": MetadataValue.md("\n".join(f"- {p.name} ({p.stat().st_size / 1e6:.1f} MB)" for p in files)),
        "total_size_mb": round(sum(p.stat().st_size for p in files) / 1e6, 1),
    })


@asset(group_name="tnbc_pipeline", deps=[raw_data],
       description="DVC stage `prepare`: labels + clinical + IHC stand-in genes -> data/processed/dataset.parquet.")
def processed_dataset(context: AssetExecutionContext) -> MaterializeResult:
    _, skipped = dvc_repro(context, "prepare")
    summary = json.loads((ROOT / "data" / "processed" / "data_summary.json").read_text())
    counts = pd.DataFrame(summary["subtype_counts"].items(), columns=["subtype", "patients"])
    return MaterializeResult(metadata={
        "n_patients": summary["n_patients"],
        "subtype_counts": md_table(counts),
        "dvc_skipped_unchanged": skipped,
    })


@asset(group_name="tnbc_pipeline", deps=[processed_dataset],
       description="DVC stage `train`: repeated CV of every feature set x model (logged to MLflow); refits and registers the final model.")
def trained_models(context: AssetExecutionContext) -> MaterializeResult:
    _, skipped = dvc_repro(context, "train")
    cv = pd.read_csv(METRICS / "cv_results.csv")
    fm = PARAMS["final_model"]
    final = cv[(cv.feature_set == fm["feature_set"]) & (cv.model == fm["model"])].iloc[0]
    return MaterializeResult(metadata={
        "cv_results": md_table(cv.sort_values("f1_macro_mean", ascending=False)),
        "final_model": f"{fm['model']} on {fm['feature_set']}",
        "final_macro_f1": float(final.f1_macro_mean),
        "final_accuracy": float(final.accuracy_mean),
        "dvc_skipped_unchanged": skipped,
    })


@asset(group_name="tnbc_pipeline", deps=[trained_models],
       description="DVC stage `evaluate`: per-subtype metrics, confusion matrix and drop-one-stain analysis.")
def model_evaluation(context: AssetExecutionContext) -> MaterializeResult:
    _, skipped = dvc_repro(context, "evaluate")
    per_class = json.loads((METRICS / "per_class.json").read_text())
    pc = pd.DataFrame(per_class).T.reset_index(names="subtype")
    ablation = pd.read_csv(METRICS / "stain_ablation.csv")
    return MaterializeResult(metadata={
        "per_subtype": md_table(pc),
        "stain_ablation": md_table(ablation),
        "confusion_matrix_plot": MetadataValue.path(str(ROOT / "plots" / "confusion_matrix.png")),
        "dvc_skipped_unchanged": skipped,
    })


@asset(group_name="tnbc_pipeline", deps=[model_evaluation],
       description="Promotes the newest registered model version to the MLflow alias `champion` if it passes the quality gate. KServe will serve models:/tnbc-subtype-classifier@champion.")
def champion_model(context: AssetExecutionContext) -> MaterializeResult:
    from mlflow import MlflowClient

    cfg = PARAMS["mlflow"]
    name = cfg["registered_model_name"]
    cv = pd.read_csv(METRICS / "cv_results.csv")
    fm = PARAMS["final_model"]
    f1 = float(cv[(cv.feature_set == fm["feature_set"]) & (cv.model == fm["model"])].f1_macro_mean.iloc[0])

    client = MlflowClient(tracking_uri=f"sqlite:///{ROOT / 'mlflow.db'}")
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"No versions of '{name}' in the MLflow registry yet; run the train stage first.")
    newest = max(versions, key=lambda v: int(v.version))

    try:
        current = client.get_model_version_by_alias(name, "champion").version
    except Exception:
        current = None

    if f1 < F1_FLOOR:
        context.log.warning(f"Version {newest.version} macro F1 {f1:.3f} < {F1_FLOOR}; champion stays at v{current}.")
        promoted = False
    else:
        client.set_registered_model_alias(name, "champion", newest.version)
        client.set_model_version_tag(name, newest.version, "cv_macro_f1", f"{f1:.4f}")
        client.set_model_version_tag(name, newest.version, "promoted_by", f"dagster run {context.run_id}")
        context.log.info(f"Promoted {name} v{newest.version} to @champion (macro F1 {f1:.3f}).")
        promoted, current = True, newest.version

    return MaterializeResult(metadata={
        "registered_model": name,
        "newest_version": int(newest.version),
        "champion_version": int(current) if current else -1,
        "promoted_this_run": promoted,
        "model_uri": f"models:/{name}@champion",
        "macro_f1": f1,
    })


@asset(group_name="tnbc_pipeline", deps=[champion_model],
       description="Exports the @champion model to serving/model/ (model.joblib + metadata.json) for the KServe prediction server. Rebuild the image with: bash serving/serve_local.sh")
def serving_model(context: AssetExecutionContext) -> MaterializeResult:
    proc = subprocess.run(["python", "serving/export_champion.py"], cwd=ROOT, capture_output=True, text=True)
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.strip():
            context.log.info(line)
    if proc.returncode != 0:
        raise RuntimeError("serving/export_champion.py failed")
    meta = json.loads((ROOT / "serving" / "model" / "metadata.json").read_text())
    return MaterializeResult(metadata={
        "model_version": meta["model_version"],
        "cv_macro_f1": meta.get("cv_macro_f1") or "",
        "features": ", ".join(meta["features"]),
        "sklearn_version": meta["sklearn_version"],
        "endpoint": "POST /v1/models/tnbc-subtype:predict",
    })


# ----------------------------------------------------------------------------- data & model quality checks
@asset_check(asset=processed_dataset, description="All four subtypes present and the patient count is what the paper reports.")
def dataset_is_complete():
    summary = json.loads((ROOT / "data" / "processed" / "data_summary.json").read_text())
    missing = [s for s in SUBTYPES if s not in summary["subtype_counts"]]
    ok = not missing and summary["n_patients"] == EXPECTED_PATIENTS
    return AssetCheckResult(passed=ok, severity=AssetCheckSeverity.WARN, metadata={
        "n_patients": summary["n_patients"], "expected": EXPECTED_PATIENTS, "missing_subtypes": str(missing)})


@asset_check(asset=processed_dataset, description="No IHC stand-in gene is missing for any patient.")
def ihc_values_present():
    data = pd.read_parquet(ROOT / "data" / "processed" / "dataset.parquet")
    ihc = [c for c in data.columns if c.startswith("IHC_")]
    n_missing = int(data[ihc].isna().sum().sum())
    return AssetCheckResult(passed=n_missing == 0, metadata={"missing_ihc_values": n_missing})


@asset_check(asset=trained_models, description=f"Quality gate: final model macro F1 >= {F1_FLOOR}.")
def final_model_meets_floor():
    cv = pd.read_csv(METRICS / "cv_results.csv")
    fm = PARAMS["final_model"]
    f1 = float(cv[(cv.feature_set == fm["feature_set"]) & (cv.model == fm["model"])].f1_macro_mean.iloc[0])
    return AssetCheckResult(passed=f1 >= F1_FLOOR, severity=AssetCheckSeverity.ERROR,
                            metadata={"macro_f1": f1, "floor": F1_FLOOR})


@asset_check(asset=trained_models, description="Sanity check: the IHC panel beats routine clinical data (the paper's main finding).")
def ihc_beats_clinical():
    cv = pd.read_csv(METRICS / "cv_results.csv")
    best = cv.groupby("feature_set").f1_macro_mean.max()
    gap = float(best.get("ihc_panel", 0) - best.get("clinical", 0))
    return AssetCheckResult(passed=gap > 0.1, severity=AssetCheckSeverity.WARN, metadata={"f1_gap": gap})


@asset_check(asset=model_evaluation, description="Every subtype has out-of-fold F1 >= 0.35 (BL2 is the usual weak spot).")
def no_subtype_left_behind():
    per_class = json.loads((METRICS / "per_class.json").read_text())
    worst = min(per_class, key=lambda s: per_class[s]["f1-score"])
    f1 = per_class[worst]["f1-score"]
    return AssetCheckResult(passed=f1 >= 0.35, severity=AssetCheckSeverity.WARN,
                            metadata={"weakest_subtype": worst, "weakest_f1": f1})


# ----------------------------------------------------------------------------- job, schedule, sensor
full_pipeline = define_asset_job("full_pipeline", selection=AssetSelection.groups("tnbc_pipeline"),
                                 description="raw data -> prepare -> train -> evaluate -> promote champion")

weekly_retrain = ScheduleDefinition(
    job=full_pipeline, cron_schedule="0 6 * * 1", execution_timezone="America/Denver",
    default_status=DefaultScheduleStatus.STOPPED,   # turn on in the UI (Automation tab)
)


def _fingerprint():
    """Hash of the files that should trigger retraining: data version + settings."""
    h = hashlib.md5()
    for f in ["data/raw.dvc", "params.yaml"]:
        h.update((ROOT / f).read_bytes())
    return h.hexdigest()


@sensor(job=full_pipeline, minimum_interval_seconds=60, default_status=DefaultSensorStatus.STOPPED,
        description="Starts the pipeline when data/raw.dvc (a new data version) or params.yaml changes.")
def retrain_on_change(context: SensorEvaluationContext):
    fp = _fingerprint()
    if context.cursor == fp:
        return SkipReason("Data version and params.yaml unchanged.")
    first_look = context.cursor is None
    context.update_cursor(fp)
    if first_look:
        return SkipReason("Recorded the current data/params fingerprint; will trigger on the next change.")
    return RunRequest(run_key=fp)


defs = Definitions(
    assets=[raw_data, processed_dataset, trained_models, model_evaluation, champion_model, serving_model],
    asset_checks=[dataset_is_complete, ihc_values_present, final_model_meets_floor,
                  ihc_beats_clinical, no_subtype_left_behind],
    jobs=[full_pipeline],
    schedules=[weekly_retrain],
    sensors=[retrain_on_change],
)
