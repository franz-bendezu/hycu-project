PYTHON ?= /workspaces/hycu-project/.venv/bin/python

.PHONY: help pipeline-components pipeline-components-prod validate-components dry-run-components

help:
	@echo "Targets:"
	@echo "  make pipeline-components   # production-gated run (components profile)"
	@echo "  make pipeline-components-prod  # alias of pipeline-components"
	@echo "  make validate-components   # schema validation only"
	@echo "  make dry-run-components    # preview commands for components pipeline"

pipeline-components:
	$(PYTHON) src/orchestration/run_pipeline.py --config configs/components_pipeline_production.yaml

pipeline-components-prod:
	$(MAKE) pipeline-components

validate-components:
	$(PYTHON) src/model/validate_dataset.py --dataset-yaml datasets/yolo_dataset_components_active.yaml

dry-run-components:
	$(PYTHON) src/orchestration/run_pipeline.py --config configs/components_pipeline_production.yaml --dry-run
