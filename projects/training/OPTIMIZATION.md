# ML Training & Optimization Guide

This document explains the configuration and optimization strategies used to reach high accuracy for furniture component detection.

## 🚀 Model Optimization

The training pipeline is optimized for identifying 21 unique classes, ranging from large structures (Cabinets) to tiny hardware (Hinges).

### Key Performance Settings

| Setting | Value | Impact |
| :--- | :--- | :--- |
| **Model** | `yolo11s.pt` | Upgraded from Nano to Small to handle the high number of classes (21). |
| **Input Size** | `1024px` | Critical for detecting small hardware like screws, hinges, and slides. |
| **Box Loss** | `7.5` | Weighted gain for bounding box accuracy, improving dimension estimation. |
| **Epochs** | `150` | Extended training budget to ensure convergence on complex features. |
| **Augmentation** | Mosaic/Mixup | High-intensity augmentation to help the model generalize from limited data. |

## 📂 Project Structure

```text
projects/training/
├── configs/            # Config-driven training runs
├── src/
│   ├── data_collection/# Scrapers and dataset preparation
│   ├── model/          # Training, validation, and export logic
│   └── orchestration/  # Automated pipeline runner
├── datasets/           # Raw and YOLO-formatted data
├── models/             # Exported artifacts (.pt, .onnx, .joblib)
└── runs/               # Experiment logs and weights
```

## 🛠️ Commands

### 1. Full Pipeline Run
The most robust way to train, which includes data preparation, validation, and production-gating:
```bash
make pipeline-components
```

### 2. Individual Training
To run only the training script manually:
```bash
python src/model/train.py \
  --dataset-yaml datasets/yolo_dataset_components.yaml \
  --yolo-model yolo11s.pt \
  --imgsz 1024 \
  --epochs 150 \
  --batch 8
```

### 3. Verification & Export
Validate that your dataset structure is correct before wasting GPU hours:
```bash
make validate-components
```

## 📈 Improving Accuracy (Next Steps)

If you find that specific parts (e.g., *telescopic_slide*) are still having low accuracy:

1.  **Add Background Images**: Include images of empty rooms or plain furniture surfaces to reduce "False Positives".
2.  **Dataset Balance**: Ensure you have at least 50-100 instances of each component. Use the scrapers in `src/data_collection/scrapers/` to find more images of specific parts.
3.  **Tuning**: For even more precision, consider switching to `yolo11m.pt` (Medium), though this requires more VRAM (approx 12GB+).

---
*Created on 2026-06-18 as part of the HYCU ML Optimization phase.*
