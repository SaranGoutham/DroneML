# DroneML – Project Walkthrough

## Phase 1–7: Dataset Exploration & Preprocessing

- Automated scanning of all 1,835 thermal UAV images (TIFF/JPG formats)
- Identified structural anomalies: 16 empty directories, 521 format/extension mismatches, date mismatches, negative altitude entries
- Detected 86 perceptual duplicate clusters (localized within flight dates — no cross-flight leakage)
- Preprocessed images: CLAHE contrast enhancement, gamma correction, TIFF→PNG conversion
- Split dataset by flight date: **train / validation / test** (250 / 154 / 117 images)
- Built `tf.data.Dataset` pipeline with augmentation (flip, rotation ±3%, zoom 10%, translation 5%, contrast 10%)

---

## Phase 8: MobileNetV3-Large Baseline Model

- Defined `src/models/mobilenetv3.py` with frozen ImageNet backbone + custom classifier head
- Architecture: `MobileNetV3-Large → GAP → BN → Dense(256) → Dropout → Dense(128) → Dropout → sigmoid`
- Metrics: accuracy, precision, recall, AUC

---

## Phase 9: MobileNetV3-Large Training

- `src/models/train.py` — baseline training pipeline (frozen backbone, Phase A only)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Dynamic class weights to handle Mine/No_Mine imbalance (220 vs 30 training images)

---

## Phase 10–11: Multi-Model Expansion (NEW)

Added **ResNet50** and **EfficientNetB0** as additional model candidates alongside MobileNetV3-Large.

### New Files Created

| File | Description |
|---|---|
| [`src/models/resnet50.py`](file:///c:/Code-Files/DroneML/src/models/resnet50.py) | ResNet50 model (175 layers, 24.2M params) with identical API to mobilenetv3.py |
| [`src/models/efficientnetb0.py`](file:///c:/Code-Files/DroneML/src/models/efficientnetb0.py) | EfficientNetB0 model (238 layers, 4.4M params) — most parameter-efficient |
| [`src/models/train_all.py`](file:///c:/Code-Files/DroneML/src/models/train_all.py) | Unified training runner — trains all 3 models sequentially |
| [`src/models/compare_models.py`](file:///c:/Code-Files/DroneML/src/models/compare_models.py) | Comparison dashboard — training curves, bar charts, ROC curves, confusion matrices, Markdown report |
| [`src/models/evaluate_models.py`](file:///c:/Code-Files/DroneML/src/models/evaluate_models.py) | Standalone test-set evaluator — reload saved models and evaluate without retraining |

### Updated Files

| File | Change |
|---|---|
| [`src/models/config.py`](file:///c:/Code-Files/DroneML/src/models/config.py) | Added ResNet50 + EfficientNetB0 hyperparameters, per-model paths, comparison directory |
| [`requirements.txt`](file:///c:/Code-Files/DroneML/requirements.txt) | Added `scikit-learn>=1.5` for ROC curves and confusion matrices |

### Architecture (All Models)

All three models follow the same classifier head pattern:

```
Input (224×224×3)
  → [Backbone — frozen ImageNet weights]
  → GlobalAveragePooling2D
  → BatchNormalization
  → Dense(256, relu) + L2(1e-4) + Dropout(0.4)
  → Dense(128, relu) + L2(1e-4) + Dropout(0.3)
  → Dense(1, sigmoid)  ← binary: Mine vs No_Mine
```

**Key difference**: ResNet50 and EfficientNetB0 include a `Rescaling(255.0)` layer before the backbone because their internal preprocessing expects `[0, 255]` while the data pipeline produces `[0, 1]`.

### Training Strategy (Per Model)

| Phase | Backbone | Learning Rate | Max Epochs | EarlyStopping |
|---|---|---|---|---|
| A – Frozen | All frozen | 1e-3 | 50 | patience=8 |
| B – Fine-tune | Top 20 layers unfrozen | 1e-5 | 20 | patience=5 |

### Model Parameter Counts

| Model | Params | Backbone Layers |
|---|---|---|
| MobileNetV3-Large | ~5.5M | 186 |
| ResNet50 | ~24.2M | 175 |
| EfficientNetB0 | ~4.4M | 238 |

---

## Phase 12: Comparison & Visualization (NEW)

### How to Run

```bash
# Step 1: Train all three models
python -m src.models.train_all

# Step 2: Generate all comparison plots and report
python -m src.models.compare_models

# (Optional) Re-evaluate saved models without retraining
python -m src.models.evaluate_models
```

### Output Files

```
results/
├── checkpoints/
│   ├── mobilenetv3_best.keras
│   ├── mobilenetv3_finetuned.keras
│   ├── resnet50_best.keras
│   ├── resnet50_finetuned.keras
│   ├── efficientnetb0_best.keras
│   └── efficientnetb0_finetuned.keras
├── models/
│   ├── mobilenetv3_final.keras
│   ├── resnet50_final.keras
│   └── efficientnetb0_final.keras
├── history/
│   ├── mobilenetv3_history.json
│   ├── resnet50_history.json
│   └── efficientnetb0_history.json
└── comparison/
    ├── comparison_results.json      ← all test metrics side-by-side
    ├── training_curves.png          ← loss & accuracy for all models
    ├── metric_comparison.png        ← bar chart: Accuracy/AUC/Precision/Recall
    ├── roc_curves.png               ← ROC/AUC curves on test set
    ├── confusion_matrix_*.png       ← one per model
    └── comparison_report.md         ← Markdown winner report
```

### Comparison Script Outputs

`compare_models.py` generates:
1. **Training curves** — loss + accuracy for all 3 models on the same axes (solid=train, dotted=val)
2. **Metric bar chart** — grouped bars for Accuracy, AUC, Precision, Recall per model
3. **ROC curves** — one curve per model with sklearn AUC label
4. **Confusion matrices** — true/false positives/negatives on the test set
5. **`comparison_report.md`** — declares the winner, full metric table, ROC AUC table

---

## Key Design Decisions

- **Backward compatibility**: `train.py` (original MobileNetV3-only script) still works unchanged — all legacy config symbols are preserved
- **TensorBoard optional**: Training runs without TensorBoard if not installed; logs gracefully skipped
- **Sequential model loading**: `train_all.py` calls `tf.keras.backend.clear_session()` between models to free GPU/CPU memory
- **Scikit-learn integration**: `compare_models.py` and `evaluate_models.py` use sklearn for ROC-AUC, confusion matrices, and classification reports (not available in Keras metrics alone)
