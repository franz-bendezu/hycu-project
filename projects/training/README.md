# Training Pipeline

This project handles data collection, model training, and export artifacts consumed by the inference service.

## Scope aligned to proposal

- Week 4: Data collection uses Requests and Selenium.
- Week 11: Model generation uses supervised CNN detection (YOLOv11) and regression for dimension estimation.

## Commands

1. Collect data and metadata:

```bash
python data_collection/collect_data.py \
	--sources-json data_collection/sources.example.json \
	--output-dir datasets/raw \
	--metadata-csv datasets/raw/metadata.csv
```

Use `data_collection/sources.example.json` as the starting template.

Real open-data collection (executed in this workspace):

```bash
python data_collection/collect_wikimedia_open.py \
	--queries-json data_collection/commons_queries.json \
	--output-dir datasets/raw/images \
	--metadata-csv datasets/raw/metadata_open.csv \
	--per-category 6 \
	--thumb-width 320
```

Resulting files:
- `datasets/raw/images/` (downloaded image files)
- `datasets/raw/metadata_open.csv` (image/license/source metadata)
- `datasets/raw/metadata_open.summary.json` (collection summary)

2. Train YOLOv11 + regression model:

```bash
python train/train.py \
	--dataset-yaml datasets/yolo_dataset.yaml \
	--regression-csv datasets/regression_features.csv \
	--yolo-model yolo11n.pt \
	--epochs 30 \
	--project-dir models
```

Use `datasets/yolo_dataset.example.yaml` and `datasets/regression_features.example.csv` as templates.

3. Export inference artifacts:

```bash
python export/export_model.py \
	--yolo-weights models/yolo11_furniture/weights/best.pt \
	--regression-model models/regression/dimension_regressor.joblib \
	--out-dir models/export
```

## Notes

- Provide your own scraping targets and respect each website's terms of service.
- Training scripts are practical MVP baselines and are intentionally modular for later scaling.
