#!/usr/bin/env bash
# Start the Dagster UI (asset graph, runs, checks, schedule, sensor) on port 3000.
# Run history is kept in .dagster/ (ignored by git).
cd "$(dirname "$0")"
export DAGSTER_HOME="$PWD/.dagster"
export MLFLOW_DISABLE_AGENT_HINT=1
mkdir -p "$DAGSTER_HOME" && touch "$DAGSTER_HOME/dagster.yaml"
dagster dev -f orchestration/definitions.py -h 0.0.0.0 -p 3000
