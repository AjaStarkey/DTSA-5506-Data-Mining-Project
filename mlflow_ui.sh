#!/usr/bin/env bash
# Start the MLflow dashboard so it works through GitHub Codespaces port forwarding.
# Codespaces forwards the page as "localhost:5000", which MLflow's security check
# must be told to allow. The port stays private to your GitHub login.
cd "$(dirname "$0")"
mlflow ui --backend-store-uri sqlite:///mlflow.db \
  --allowed-hosts "localhost,localhost:5000,127.0.0.1,127.0.0.1:5000" \
  --cors-allowed-origins "https://${CODESPACE_NAME}-5000.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
