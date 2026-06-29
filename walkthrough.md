# DroneML – Project Walkthrough

## Phase 1–7: Dataset Exploration & Preprocessing
- Automated scanning of all 1,835 thermal UAV images (TIFF/JPG formats).
- Identified structural anomalies: 16 empty directories, 521 format/extension mismatches, date mismatches, negative altitude entries.
- Detected 86 perceptual duplicate clusters (localized within flight dates — no cross-flight leakage).
- Preprocessed images: CLAHE contrast enhancement, gamma correction, TIFF→PNG conversion.
- Split dataset by flight date: **train / validation / test** (250 / 154 / 117 images).
- Built `tf.data.Dataset` pipeline with augmentation (flip, rotation ±3%, zoom 10%, translation 5%, contrast 10%).

---

## Phase 8: Baseline Models (MobileNetV3, ResNet50, EfficientNetB0)
- Configured three deep backbones with custom binary classification heads:
  - **MobileNetV3-Large** (~5.5M parameters)
  - **ResNet50** (~24.2M parameters)
  - **EfficientNetB0** (~4.4M parameters)
- Architecture style: `Backbone → GAP → BatchNormalization → Dense(256, relu) + L2 + Dropout → Dense(128, relu) + L2 + Dropout → Dense(1, sigmoid)`.
- Internal rescaling (`Rescaling(255.0)`) integrated inside ResNet50 and EfficientNetB0 to match data pipeline normalization (`[0, 1]` to `[0, 255]`).

---

## Phase 9: Fine-Tuning Pipeline
Fine-tuning is handled in a two-stage manner to maximize stability, with support for standalone runs:
- `src/models/train_all.py` — sequences frozen-backbone training (Phase A) and optional fine-tuning for all models.
- [`src/models/finetune.py`](file:///c:/Code-Files/DroneML/src/models/finetune.py) [NEW] — Standalone fine-tuning runner. Loads the best frozen-backbone checkpoint, unfreezes the top 20 layers of the backbone, recompiles with a lower learning rate ($10^{-5}$), and continues training with early stopping.

---

## Phase 10: Evaluation & Performance Metrics
The evaluation suite has been structured under a dedicated `src/evaluation` package to isolate metrics and plotting logic:
- [`src/evaluation/metrics.py`](file:///c:/Code-Files/DroneML/src/evaluation/metrics.py) [NEW] — Comprehensive evaluation runner. Computes:
  - Accuracy, Precision, Recall, and Loss.
  - F1-Score and ROC-AUC (using scikit-learn).
  - Confusion Matrix plot annotated with percentages (`results/evaluation/confusion_matrix_<model>.png`).
  - Classification Report text printout.
  - Combined ROC/AUC Curve plot (`results/evaluation/roc_curve.png`).
  - Saved outputs: JSON metric summaries (`results/evaluation/metrics.json` and per-model files).
- [`src/evaluation/evaluate_models.py`](file:///c:/Code-Files/DroneML/src/evaluation/evaluate_models.py) [RELOCATED] — Lightweight evaluator moved from `src/models/` to `src/evaluation/` that validates model test-set performance and updates `comparison_results.json`.

---

## Phase 11: Explainability (Grad-CAM)
- [`src/evaluation/gradcam.py`](file:///c:/Code-Files/DroneML/src/evaluation/gradcam.py) [NEW] — Computes Gradient-weighted Class Activation Maps (Grad-CAM) to visualize model feature focus:
  - Targeted conv layers: `Conv_1` (MobileNetV3), `conv5_block3_out` (ResNet50), `top_activation` (EfficientNetB0).
  - Outputs: 3-panel figures showing the original image, jet colormap heatmap, and alpha-blended overlay.
  - Stratified outputs: Saved under `results/gradcam/<model>/` categorized into `mine`, `no_mine`, `fp` (False Positives), and `fn` (False Negatives).
  - Overlay Grid: Generates a unified multi-model overlay comparison grid (`results/gradcam/overlay_grid.png`).

---

## Phase 12: Error Analysis
- [`src/evaluation/error_analysis.py`](file:///c:/Code-Files/DroneML/src/evaluation/error_analysis.py) [NEW] — Stratified performance evaluation across UAV flight metadata:
  - Matches test predictions to metadata in `results/metadata_checked.csv`.
  - Saves individual FP and FN image panels in `results/error_analysis/false_positives/` and `results/error_analysis/false_negatives/`.
  - Calculates accuracy and generates bar charts stratified by:
    - Flight Date (`results/error_analysis/flight_wise_accuracy.png`)
    - Altitude Bin (`results/error_analysis/altitude_wise_accuracy.png`)
    - Mine Burial Depth (`results/error_analysis/depth_wise_accuracy.png`)
    - Zone (`results/error_analysis/zone_wise_accuracy.png`)
  - Outputs a narrative Markdown error report at `results/error_analysis/error_report.md`.

---

## Restructuring & Clean Up
- Deleted deprecated duplicate scripts `src/models/plot_training.py` and `src/models/evaluate_models.py` to prevent redundancy.
- Created [`src/evaluation/plot_training.py`](file:///c:/Code-Files/DroneML/src/evaluation/plot_training.py) [NEW] — Generates combined/individual training history plots.
- Unified output paths under `src/models/config.py`.

### How to Run Evaluation & Analysis (Once Models are Trained)

```bash
# Plot training histories
python -m src.evaluation.plot_training

# Run Phase 10 Metrics and generate confusion matrices + ROC curves
python -m src.evaluation.metrics

# Run Phase 11 Grad-CAM to visualize model predictions
python -m src.evaluation.gradcam

# Run Phase 12 Error Analysis stratified by flight/altitude/depth/zone
python -m src.evaluation.error_analysis
```
