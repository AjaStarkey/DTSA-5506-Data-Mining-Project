"""KServe custom predictor for the TNBC subtype classifier.

Runs the same way on a laptop, in Docker, or as a KServe InferenceService:
    python predictor.py --model_name tnbc-subtype --model_dir serving/model

Request (KServe V1 protocol), POST /v1/models/tnbc-subtype:predict
    {"instances": [{"AR": 7.86, "FOXA1": 12.30, "Ki67": 11.38, "CK5": 10.38,
                    "CK6": 6.15, "EGFR": 9.73, "Vimentin": 13.81}]}
Each value is the stain's stand-in on the training scale: log2(RSEM + 1) mRNA of the
matching gene. A value may be null (it is then imputed with the training median).
Instances may also be plain lists of 7 numbers in the order given by GET /v1/models/tnbc-subtype/metadata...
(see "features" in serving/model/metadata.json).

Response
    {"predictions": [{"subtype": "LAR", "probabilities": {"BL1": 0.03, ...}}],
     "model": {"name": ..., "version": ..., "cv_macro_f1": ...}}
"""
import argparse
import json
import logging
import math
from pathlib import Path

import joblib
import pandas as pd
from kserve import Model, ModelServer, model_server
from kserve.errors import InvalidInput

# Accepted input names (case-insensitive) -> model column
ALIASES = {
    "ar": "IHC_AR", "ihc_ar": "IHC_AR",
    "foxa1": "IHC_FOXA1", "ihc_foxa1": "IHC_FOXA1",
    "ki67": "IHC_MKI67", "ki-67": "IHC_MKI67", "mki67": "IHC_MKI67", "ihc_mki67": "IHC_MKI67",
    "ck5": "IHC_KRT5", "krt5": "IHC_KRT5", "ihc_krt5": "IHC_KRT5",
    "ck6": "IHC_KRT6A", "krt6a": "IHC_KRT6A", "ihc_krt6a": "IHC_KRT6A",
    "egfr": "IHC_EGFR", "ihc_egfr": "IHC_EGFR",
    "vimentin": "IHC_VIM", "vim": "IHC_VIM", "ihc_vim": "IHC_VIM",
}


class TNBCSubtypeModel(Model):
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.model_dir = Path(model_dir)
        self.model = None
        self.meta = {}
        self.load()

    def load(self):
        self.model = joblib.load(self.model_dir / "model.joblib")
        self.meta = json.loads((self.model_dir / "metadata.json").read_text())
        self.features = self.meta["features"]
        self.ready = True
        logging.info("Loaded %s v%s", self.meta.get("model_name"), self.meta.get("model_version"))
        return self.ready

    def _row(self, inst, i):
        if isinstance(inst, dict):
            row, unknown = {}, []
            for key, value in inst.items():
                col = ALIASES.get(str(key).strip().lower())
                if col is None:
                    unknown.append(key)
                else:
                    row[col] = value
            missing = [f for f in self.features if f not in row]
            if unknown or missing:
                raise InvalidInput(f"instance {i}: unknown fields {unknown}, missing fields {missing}. "
                                   f"Expected: AR, FOXA1, Ki67, CK5, CK6, EGFR, Vimentin")
            values = [row[f] for f in self.features]
        elif isinstance(inst, (list, tuple)):
            if len(inst) != len(self.features):
                raise InvalidInput(f"instance {i}: expected {len(self.features)} values in the order {self.features}")
            values = list(inst)
        else:
            raise InvalidInput(f"instance {i}: must be an object or a list of {len(self.features)} numbers")
        clean = []
        for f, v in zip(self.features, values):
            if v is None:
                clean.append(math.nan)
                continue
            try:
                clean.append(float(v))
            except (TypeError, ValueError):
                raise InvalidInput(f"instance {i}: {f} must be a number or null, got {v!r}")
        return clean

    def predict(self, payload, headers=None):
        if isinstance(payload, (bytes, str)):          # body sent without a JSON content type
            try:
                payload = json.loads(payload)
            except ValueError:
                raise InvalidInput("Request body is not valid JSON")
        if not isinstance(payload, dict) or "instances" not in payload:
            raise InvalidInput('Request body must be {"instances": [...]} (KServe V1 protocol)')
        rows = [self._row(inst, i) for i, inst in enumerate(payload["instances"])]
        X = pd.DataFrame(rows, columns=self.features)
        proba = self.model.predict_proba(X)
        classes = [str(c) for c in self.model.classes_]
        preds = []
        for p in proba:
            best = int(p.argmax())
            preds.append({"subtype": classes[best],
                          "probabilities": {c: round(float(v), 3) for c, v in zip(classes, p)}})
        return {"predictions": preds,
                "model": {k: self.meta.get(k) for k in ("model_name", "model_version", "cv_macro_f1")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[model_server.parser])
    parser.add_argument("--model_dir", default="/mnt/models", help="folder with model.joblib and metadata.json")
    args, _ = parser.parse_known_args()
    ModelServer().start([TNBCSubtypeModel(args.model_name, args.model_dir)])
