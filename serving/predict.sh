#!/usr/bin/env bash
# Send the four example patients in sample_request.json to the running server.
# Usage: bash serving/predict.sh [host:port]   (default localhost:8080)
cd "$(dirname "$0")"
HOST="${1:-localhost:8080}"
curl -s -X POST "http://${HOST}/v1/models/tnbc-subtype:predict" \
  -H "Content-Type: application/json" -d @sample_request.json | python3 -m json.tool
