# Training Pipeline

This project handles data collection, model training, and export artifacts consumed by the inference service.

## Standardized structure

```text
training/
	configs/
		components_pipeline_production.yaml
	datasets/
	models/
	runs/
	scripts/
	src/
		data_collection/
			prepare_yolo_dataset.py
			scrapers/
		model/
			train.py
			validate_dataset.py
			export_model.py
		orchestration/
			run_pipeline.py
	Makefile
	requirements.txt
```

Key best-practice principles used:
- config-driven runs (`configs/*.yaml`)
- dataset quality gate before training (`src/model/validate_dataset.py`)
- reproducible automation (`src/orchestration/run_pipeline.py`)
- per-run manifest and latest-run marker for traceability
- production-gated mode with metric thresholds and artifact checksums

## Scope aligned to proposal

- Week 4: Data collection uses Requests and Selenium.
- Week 11: Model generation uses supervised CNN detection (YOLOv11) and regression for dimension estimation.

## Commands

1. Production Pipeline (Recommended):

```bash
# Full automated run (Prepare -> Validate -> Train -> Export -> Manifest)
bash scripts/components_pipeline.sh run

# Or via Makefile
make pipeline-components
```

2. Collect data and metadata (Scrapers):

```bash
python src/data_collection/scrapers/collect_data.py \
	--sources-json src/data_collection/scrapers/sources.example.json \
	--output-dir datasets/raw \
	--metadata-csv datasets/raw/metadata.csv
```

Real open-data collection (executed in this workspace):

```bash
python src/data_collection/scrapers/collect_wikimedia_open.py \
	--queries-json src/data_collection/scrapers/commons_queries.json \
	--output-dir datasets/raw/images \
	--metadata-csv datasets/raw/metadata_open.csv \
	--per-category 6 \
	--thumb-width 320
```

3. Train YOLOv11 + regression model (Manual):

```bash
python src/model/train.py \
	--dataset-yaml datasets/yolo_dataset.yaml \
	--regression-csv datasets/regression_features.csv \
	--yolo-model yolo11n.pt \
	--epochs 30 \
	--project-dir runs/train \
	--regression-out-dir models/regression
```

The default training output now writes YOLO runs under `runs/train/` inside this
`projects/training` folder (instead of generic external `runs/` locations).

Standardized automation runner (recommended for repeatable ML workflow):

```bash
python train/run_pipeline.py \
	--config configs/components_pipeline_production.yaml \
	--prepare-dataset \
	--prepare-class-profile components \
	--dataset-yaml datasets/yolo_dataset_components.yaml \
	--yolo-model yolo11n.pt \
	--epochs 40 \
	--batch 16 \
	--workers 8 \
	--project-dir runs/train \
	--deploy-to-inference
```

What this runner automates:
- optional dataset preparation
- detector training (with automatic `--workers 0` retry when shared-memory errors happen)
- ONNX export
- deploy model to inference service models path
- run metadata in `runs/train/<experiment>/pipeline_manifest.json`
- latest run marker in `runs/train/LATEST_RUN.txt`

Production mode (stricter gates):

```bash
python src/orchestration/run_pipeline.py --config configs/components_pipeline_production.yaml
```

When `production_mode: true`, the pipeline additionally enforces:
- minimum train/val sample counts
- maximum class imbalance ratio
- promotion thresholds from training metrics (`mAP50`, `mAP50-95`, `precision`, `recall`)
- artifact metadata with file size and `sha256` checksums in the manifest

Makefile shortcuts:

```bash
make pipeline-components
make pipeline-components-prod
make validate-components
make dry-run-components
```

Bash script shortcut (from `projects/training`):

```bash
./scripts/components_pipeline.sh run
./scripts/components_pipeline.sh dry-run
./scripts/components_pipeline.sh validate
./scripts/components_pipeline.sh prepare
./scripts/components_pipeline.sh help
```

By default, the script uses:
- config: `configs/components_pipeline_production.yaml`
- python: `/workspaces/hycu-project/.venv/bin/python` (if present)

Optional overrides:

```bash
CONFIG_PATH=configs/components_pipeline_production.yaml ./scripts/components_pipeline.sh run
PYTHON_BIN=python3 ./scripts/components_pipeline.sh dry-run
```

The recommended daily command for component model iteration is:

```bash
make pipeline-components
```

Use `datasets/yolo_dataset.example.yaml` and `datasets/regression_features.example.csv` as templates.

Labeling preparation before training (build train/val folders + queue CSV + empty `.txt` labels):

```bash
python src/data_collection/prepare_yolo_dataset.py \
	--raw-root datasets/raw/images \
	--out-root datasets/yolo \
	--val-ratio 0.2 \
	--mode copy \
	--clean
```

For component-level detection classes (door panel, side panel, hinge, slides, etc.),
generate the component class YAML profile:

```bash
python src/data_collection/prepare_yolo_dataset.py \
	--raw-root datasets/raw/images \
	--out-root datasets/yolo \
	--val-ratio 0.2 \
	--mode copy \
	--class-profile components
```

Then annotate files listed in `datasets/yolo/labeling_queue.csv` and save YOLO boxes into matching paths under:
- `datasets/yolo/labels/train/*.txt`
- `datasets/yolo/labels/val/*.txt`

Generated dataset YAML for training is written to `datasets/yolo_dataset.yaml`.
For component profile it is written to `datasets/yolo_dataset_components.yaml`.

Component detection training rerun (outputs remain inside `projects/training/runs/train`):

```bash
python src/model/train.py \
	--dataset-yaml datasets/yolo_dataset_components.yaml \
	--yolo-model yolo11n.pt \
	--epochs 40 \
	--project-dir runs/train \
	--experiment furniture_components
```

3. Export inference artifacts:

```bash
python export/export_model.py \
	--yolo-weights runs/train/yolo11_furniture/weights/best.pt \
	--regression-model models/regression/dimension_regressor.joblib \
	--out-dir models/export
```

For production-like repeated runs, prefer `train/run_pipeline.py` instead of manually
calling train then copy/deploy steps.

## Notes

- Provide your own scraping targets and respect each website's terms of service.
- Training scripts are practical MVP baselines and are intentionally modular for later scaling.
