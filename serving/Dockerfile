# Prediction server for the TNBC subtype classifier (KServe custom predictor).
# Build from the repo root with:  bash serving/serve_local.sh
FROM python:3.13-slim

# Match the scikit-learn and numpy versions the model was saved with (read from metadata.json)
ARG SKLEARN_VERSION
ARG NUMPY_VERSION
RUN pip install --no-cache-dir "kserve==0.21.0" "scikit-learn==${SKLEARN_VERSION}" "numpy==${NUMPY_VERSION}" joblib

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY predictor.py .
COPY model/ /mnt/models/
USER 1000

EXPOSE 8080
ENTRYPOINT ["python", "predictor.py", "--model_name", "tnbc-subtype", "--model_dir", "/mnt/models", "--http_port", "8080"]
