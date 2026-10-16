"""Export the MLflow @champion model into serving/model/ for the prediction server.

Writes:
  serving/model/model.joblib     the fitted scikit-learn pipeline
  serving/model/metadata.json    model version, CV score, input features, classes, library versions
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy
import sklearn
import yaml
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "serving" / "model"


def main():
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    name = params["mlflow"]["registered_model_name"]
    uri = f"sqlite:///{ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    version = client.get_model_version_by_alias(name, "champion")
    model = mlflow.sklearn.load_model(f"models:/{name}@champion")

    features = list(model.feature_names_in_)
    gene_to_stain = {"AR": "AR", "FOXA1": "FOXA1", "MKI67": "Ki-67", "KRT5": "CK5/6 (CK5)",
                     "KRT6A": "CK5/6 (CK6)", "EGFR": "EGFR", "VIM": "Vimentin"}
    meta = {
        "model_name": name,
        "model_version": int(version.version),
        "mlflow_run_id": version.run_id,
        "cv_macro_f1": version.tags.get("cv_macro_f1"),
        "final_model": params["final_model"],
        "features": features,
        "feature_descriptions": {f: f"{gene_to_stain.get(f.replace('IHC_', ''), f)} stand-in: "
                                    f"log2(RSEM + 1) mRNA of {f.replace('IHC_', '')}" for f in features},
        "classes": [str(c) for c in model.classes_],
        "sklearn_version": sklearn.__version__,
        "numpy_version": numpy.__version__,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT / "model.joblib")
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Exported {name} v{meta['model_version']} (@champion) -> {OUT}")
    print(f"  features: {', '.join(features)}")
    print(f"  scikit-learn {meta['sklearn_version']}, numpy {meta['numpy_version']}")


if __name__ == "__main__":
    main()
