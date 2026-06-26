# Walkthrough of Thermal Image Dataset Exploration

This document walks through the exploratory analysis performed on the thermal image dataset in `data/mendeley`.

## Accomplished Work

1. **Automated Analysis Script**: Created and refined [scan_dataset.py](file:///C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/scan_dataset.py) to:
   - Perform recursive structural scanning of directories.
   - Verify magic bytes for format verification (checking actual vs extension).
   - Compute dimensions, channels, dHashes, MD5, and pixel statistics.
   - Extract EXIF and XMP flight metadata tags (GPS position, absolute and relative flight altitudes, gimbal and flight pitch/roll/yaw, band name, central wavelength).
   - Assess image quality metrics (Laplacian blur score, clipping percentages, contrast standard deviations).
   - Generate representative side-by-side visualization panels showing original image, histogram, properties, and telemetry.
2. **Analysis Execution**: Executed the script successfully to parse all 1,835 files in the dataset.
3. **Data Compilation**: Exported results to `scan_results.json` and `summary_stats.json` for review.
4. **Final Research Summary Report**: Prepared a detailed summary in [research_report.md](file:///C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/research_report.md) presenting our findings, carousels of visualizations, critical anomalies, and concrete prioritized recommendations for machine learning model training.

## Key Verification Results & Discoveries

### 1. Structural Anomalies
* **Empty Directories**: Identified 16 empty folders:
  - `13_02_2020/JPG/min1_1m` to `min7_1m` (7 directories)
  - `20_02_2020/TIFF/Zone 1` to `Zone 9` (9 directories - meaning flight date `20_02_2020` has no TIFF files).
* **Format-Extension Mismatch**: 521 files inside the `TIFF` subfolders have the `.jpg` extension but start with `II*\x00` (little-endian TIFF).
* **Date Mismatch**: Flight folder `03_03_2020` contains images with EXIF timestamps showing they were captured on March 1, 2020.
* **Relative Altitude Errors**: Several filenames under `03_03_2020` and `24_02_2020` have negative altitudes (e.g. `FlightAltitude__-1.m`), indicating sensor zeroing/drift issues.

### 2. Paired Filename Hazard
* Files sharing identical filenames across format folders (e.g., `FlightAltitude__2.2m_TemperatureInCenter__31°C.jpg`) were discovered to have **different EXIF capture timestamps** (up to 20 seconds apart) and low perceptual similarity. 
* This is caused by parameter rounding. Users should **never** pair JPG and TIFF images by filename directly, as this will lead to misaligned inputs.

### 3. Duplicate and Leakage Integrity
* Perceptual similarity clustering (Hamming dHash distance <= 3) found **86 clusters** containing consecutive frames.
* All clusters were strictly localized within their respective flight dates and formats. No cross-flight leakage was found, confirming that **split-by-date** is a robust validation strategy.

## File Artifacts Created
* Analysis script: [scan_dataset.py](file:///C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/scan_dataset.py)
* Detailed Research Report: [research_report.md](file:///C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/research_report.md)
* Raw output results: [scan_results.json](file:///C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/scan_results.json)
* Aggregated summary statistics: [summary_stats.json](file:///C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/summary_stats.json)
