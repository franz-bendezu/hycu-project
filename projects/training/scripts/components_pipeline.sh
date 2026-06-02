#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${TRAINING_DIR}/../.." && pwd)"

DEFAULT_CONFIG="configs/components_pipeline_production.yaml"
CONFIG_PATH="${CONFIG_PATH:-$DEFAULT_CONFIG}"

if [[ -x "${WORKSPACE_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${WORKSPACE_DIR}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

MODE="${1:-run}"

run_pipeline() {
  local extra_args=("$@")
  cd "${TRAINING_DIR}"
  "${PYTHON_BIN}" src/orchestration/run_pipeline.py --config "${CONFIG_PATH}" "${extra_args[@]}"
}

case "$MODE" in
  run)
    run_pipeline
    ;;
  dry-run)
    run_pipeline --dry-run
    ;;
  validate)
    cd "${TRAINING_DIR}"
    "${PYTHON_BIN}" src/model/validate_dataset.py --dataset-yaml datasets/yolo_dataset_components.yaml
    ;;
  prepare)
    cd "${TRAINING_DIR}"
    "${PYTHON_BIN}" src/data_collection/prepare_yolo_dataset.py \
      --raw-root datasets/raw/images \
      --out-root datasets/yolo \
      --val-ratio 0.2 \
      --mode copy \
      --class-profile components
    ;;
  help|-h|--help)
    cat <<'EOF'
Usage: scripts/components_pipeline.sh [mode]

Modes:
  run       Run components production pipeline (default)
  dry-run   Preview full pipeline command execution
  validate  Validate labels against components dataset YAML
  prepare   Rebuild staged YOLO dataset for components profile
  help      Show this message

Optional environment variables:
  CONFIG_PATH   Override config path (default: configs/components_pipeline_production.yaml)
  PYTHON_BIN    Override python executable
EOF
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Run scripts/components_pipeline.sh help" >&2
    exit 2
    ;;
esac
