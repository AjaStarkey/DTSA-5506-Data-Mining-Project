#!/usr/bin/env bash
# Export the @champion model, build the Docker image, start the server on port 8080 and test it.
set -e
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1

echo "== 1/4  Exporting the @champion model from MLflow =="
python serving/export_champion.py
SKLEARN_VERSION=$(python -c "import json; print(json.load(open('serving/model/metadata.json'))['sklearn_version'])")
NUMPY_VERSION=$(python -c "import json; print(json.load(open('serving/model/metadata.json'))['numpy_version'])")
TAG=$(python -c "import json; print('v' + str(json.load(open('serving/model/metadata.json'))['model_version']))")

echo "== 2/4  Building image tnbc-subtype:${TAG} (first build takes a few minutes) =="
docker build -t "tnbc-subtype:${TAG}" -t tnbc-subtype:latest \
  --build-arg SKLEARN_VERSION="$SKLEARN_VERSION" --build-arg NUMPY_VERSION="$NUMPY_VERSION" serving

echo "== 3/4  Starting the server on port 8080 =="
docker rm -f tnbc-subtype >/dev/null 2>&1 || true
docker run -d --name tnbc-subtype -p 8080:8080 tnbc-subtype:latest >/dev/null
for i in $(seq 1 60); do
  curl -s localhost:8080/v1/models/tnbc-subtype | grep -q '"ready":true' && break
  sleep 1
done
curl -s localhost:8080/v1/models/tnbc-subtype; echo

echo "== 4/4  Test request: one typical patient of each subtype (BL1, BL2, LAR, M) =="
bash serving/predict.sh
echo
echo "Server is running. Stop it with:  docker stop tnbc-subtype"
