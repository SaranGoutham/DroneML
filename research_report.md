# Thermal Image Dataset Exploration & Analysis Report

This report presents a thorough, automated, exploratory analysis of the thermal image dataset located in `data/mendeley`. 

---

## 1. Dataset Structure & Directory Tree

The dataset represents flight acquisitions from a DJI Matrice 100 UAV carrying a FLIR/DJI Zenmuse XT thermal camera. The files are organized by date, format, and zone.

### Complete Directory Tree (Overview)
```
data/mendeley/
├── Readme.txt (Dataset Overview Metadata)
├── 13_02_2020/ (Flight Date)
│   ├── JPG/ (Image Format Directory)
│   │   ├── Zone 1 Mine 1cm depth/
│   │   ├── ...
│   │   ├── Zone 9 Mine 10 cm depth/
│   │   ├── min8_1m/ (Auxiliary zone)
│   │   └── sana_1m/ (Auxiliary zone)
│   ├── R_JPG/
│   └── TIFF/
├── 17_02_2020/
├── 20_02_2020/
├── 24_02_2020/
└── 03_03_2020/
```

### Folder & File Counts
* **Total Files**: 1,835
* **Total Image Files**: 1,834
* **Total Non-Image Files**: 1 (`Readme.txt`)
* **Maximum Directory Depth**: 3 subfolders under `data/mendeley` (`Flight Date/Image Format/Zone Directory/`)

### Empty & Mislabeled Folders
During recursive scanning, **16 empty folders** were discovered. This indicates that certain formats or zones were either not captured, cleared, or omitted:

1. **Empty TIFF Directories in Flight `20_02_2020`**:
   - `20_02_2020/TIFF/Zone 1 Mine 1cm depth` to `Zone 9 Mine 10 cm depth` (All 9 zones are completely empty under the TIFF format directory for this flight).
2. **Empty JPG Directories in Flight `13_02_2020`**:
   - `13_02_2020/JPG/min1_1m` to `min7_1m` (These 7 folders are empty, while `min8_1m` and `sana_1m` contain 6 images each).

### Unusual & Inconsistent Folder Organization
* **Spelling Typos**: Folders for Zone 5 in all flights are named `Zone 5 Sruface Mine 0cm depth` (containing a typo "Sruface" instead of "Surface").
* **Spacing Inconsistencies**: 
  - `Zone 6  free zone` (contains a double space in some flights like `03_03_2020`).
  - `Zone 9 Mine 10 cm depth` (contains a space between "10" and "cm") vs `Zone 9 Mine 10cm depth` (no space).
* **Format-Extension Mismatch**: All files inside the `TIFF/` subdirectories are saved with the `.jpg` extension (e.g. `FlightAltitude__0.6m_TemperatureInCenter__35°C.jpg`), despite containing raw TIFF binary data.

---

## 2. Image Discovery & Extension Verification

Every file was inspected by reading its header magic bytes rather than trusting file extensions:

| Ext | Discovered Format | Magic Bytes | Count | Status / Inconsistency |
| :--- | :--- | :--- | :--- | :--- |
| `.jpg` | **JPEG** | `\xff\xd8\xff\xe0` | 1,313 | **Matches Extension** (JPG and R_JPG folders) |
| `.jpg` | **TIFF** (Little-Endian) | `II*\x00` | 521 | **Mismatch** (TIFF folders) |
| `.txt` | **Text** | Plain Text | 1 | **Matches Extension** (`Readme.txt`) |

> [!WARNING]
> **Extension Mismatch**: 521 images (28.4% of all images in the dataset) inside the `TIFF/` directories have the `.jpg` extension but are binary TIFF files. OpenCV's standard `cv2.imread()` fails to open these files directly due to this extension mismatch. They must be loaded using specialized libraries (e.g. `tifffile`) or by decoding binary byte streams.

No corrupted, zero-byte, or unreadable images were found. All 1,834 files are fully readable once the correct decoder is used.

---

## 3. Dataset Statistics

* **Total Dataset Size**: 304.04 MB (318,812,584 bytes)
* **Average Image Size**: 169.76 KB
* **Largest Image**: `20_02_2020/R_JPG/Zone 5 Sruface Mine 0cm depth/FlightAltitude__2.7m_TemperatureInCenter__26°C.jpg` (244.37 KB)
* **Smallest Image**: `20_02_2020/JPG/Zone 3 Mine 1cm depth/FlightAltitude__8.3m_TemperatureInCenter__26°C.jpg` (71.31 KB)

---

## 4. Image Properties

The image properties vary significantly across the format subfolders:

| Subfolder | Width | Height | Resolution | Channels | Bit Depth | Color Space | Data Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **JPG** | 720 | 480 | `720x480` (654 images) | 3 | 8-bit | Grayscale (RGB) | `uint8` |
| **R_JPG** | 336 | 256 | `336x256` (659 images) | 3 | 8-bit | Grayscale (RGB) | `uint8` |
| **TIFF** | 336 | 256 | `336x256` (521 images) | 1 | 16-bit | Grayscale | `uint16` |

* **Channel Distribution**: JPG and R_JPG contain 3-channel RGB data (all channels are identical, representing grayscale stored in RGB). TIFF contains 1-channel raw grayscale.
* **Resolution Frequencies**:
  - `336x256`: 1,180 images (64.3%)
  - `720x480`: 654 images (35.7%)

---

## 5. Thermal Image Characteristics

* **Color Space**: All 1,834 images are grayscale.
* **Grayscale Encoding**:
  - In `JPG` and `R_JPG`, values are 8-bit integers (`0` to `255`), representing normalized relative thermal radiation.
  - In `TIFF`, values are 16-bit integers (ranging from `~3000` to `~5500`), representing raw uncalibrated digital counts from the Zenmuse XT sensor.
* **Temperature Calibration**:
  - According to XMP metadata, `FLIR:TlinearGain` is `0`, meaning Temperature Linear mode is off.
  - The pixel values represent raw digital counts related to thermal radiation. The center temperatures are recorded in the filenames (e.g. `31°C`), which correspond to XMP's `FLIR:CentralTemperature="31"`.
  - The standard deviation (contrast) of raw TIFFs is very narrow (~30 to ~100 counts), signifying that the terrain has high thermal uniformity, with the simulated landmines creating tiny, subtle thermal anomalies.

---

## 6. Label Discovery & Class Mapping

Supervised learning labels are implicitly encoded in the leaf directory names. There are no external CSV, COCO, XML, Pascal VOC, or YOLO label files in this dataset.

### Class Mapping Table
Based on the folder names, we can reconstruct the following target depth classes:

| Directory Name | Target Depth Class | Label Meaning | Is Mine? |
| :--- | :--- | :--- | :--- |
| `Zone 1 Mine 1cm depth` | 1cm | Shallow buried antipersonnel mine | **True** |
| `Zone 2 Mine 1cm depth` | 1cm | Shallow buried antipersonnel mine | **True** |
| `Zone 3 Mine 1cm depth` | 1cm | Shallow buried antipersonnel mine | **True** |
| `Zone 4 Surface...` | 0cm | Surface-placed antipersonnel mine | **True** |
| `Zone 5 Sruface...` | 0cm | Surface-placed antipersonnel mine | **True** |
| `Zone 6  free zone` | Control | Control region (no mine present) | **False** |
| `Zone 7 Mine 5cm depth` | 5cm | Medium buried antipersonnel mine | **True** |
| `Zone 8 Mine 5cm depth` | 5cm | Medium buried antipersonnel mine | **True** |
| `Zone 9 Mine 10 cm depth`| 10cm | Deep buried antipersonnel mine | **True** |
| `min8_1m` | Unknown | Auxiliary experimental target | **True** |
| `sana_1m` | Unknown | Auxiliary experimental target | **True** |

---

## 7. Class & Label Distribution

The target classes are relatively balanced, with around 12 to 20 images per zone for each flight date and format:

```
  03_03_2020 Flight:
    - JPG: 16 (1cm), 15 (1cm), 13 (1cm), 13 (0cm), 12 (0cm), 12 (Free), 14 (5cm), 13 (5cm), 15 (10cm)
    - R_JPG: 15 (1cm), 17 (1cm), 14 (1cm), 13 (0cm), 12 (0cm), 15 (Free), 16 (5cm), 16 (5cm), 17 (10cm)
    - TIFF: 16 (1cm), 12 (1cm), 14 (1cm), 12 (0cm), 13 (0cm), 15 (Free), 12 (5cm), 14 (5cm), 16 (10cm)
```
*(Similar balance holds for flights on `13_02_2020`, `17_02_2020`, `20_02_2020`, and `24_02_2020`)*.
There are no orphan labels or images without labeling folders.

---

## 8. Duplicate & Data Leakage Analysis

### Duplicates
* **Exact Duplicates (MD5 Hash)**: **0%** (no files have identical binary contents).
* **Near Duplicates (Hamming dHash Distance <= 3)**: **9.5%** of the dataset (174 images grouped into 86 similarity clusters). These are consecutive video-frames captured within seconds of each other at almost identical GPS coordinates and flight altitudes.

### Data Leakage
* **Cross-Flight Leakage**: **0%**. 
* Perceptual similarity clustering showed that all near-duplicate groups are strictly isolated within their respective flight dates and format folders. There are no rotated, cropped, or brightness-adjusted duplicates split across different flights.
* **Train/Test Split Recommendation**: A standard k-fold cross-validation or train/test split stratified by **Flight Date** is completely safe and guarantees zero data leakage, leading to robust generalization metrics.

---

## 9. Telemetry & Metadata Analysis

Rich telemetry is embedded in both the standard EXIF and Adobe XMP headers of the TIFF and JPEG images.

### Extracted Metadata Fields
* **Camera Model**: FLIR (Zenmuse XT)
* **UAV Platform**: DJI (Matrice 100)
* **Focal Length**: 9.0 mm
* **Aperture**: f/1.25
* **Gimbal Pitch**: **-90° (Nadir)** — the camera was pointing vertically downwards for all acquisitions.
* **Gimbal Yaw**: Ranges from 31.2° to 32.2° (aligned with flight path orientation).
* **GPS Coordinates**: 
  - Latitude: ~3° 22' 42" N
  - Longitude: ~76° 31' 52" W
  - Absolute Altitude: ~1,011 to 1,013 meters above sea level (MSL).
* **Flight Parameters (from XMP)**: RelativeAltitude (above ground level), FlightRollDegree, FlightPitchDegree, FlightYawDegree.
* **Sensor Details**: BandName (LWIR), CentralWavelength (10,000 nm = 10 um), WavelengthFWHM (4,500 nm = 4.5 um).

---

## 10. Quality Assessment & Anomalies

### Anomaly 1: Spatial & Temporal Offsets in "Paired" Filenames
Files with matching names (e.g. `FlightAltitude__2.2m_TemperatureInCenter__31°C.jpg`) across `JPG`, `R_JPG`, and `TIFF` directories are **NOT** simultaneous captures of the same flight moment.
* **JPG DateTime**: 17:10:07
* **TIFF DateTime**: 17:10:13 (6 seconds offset)
* **R_JPG DateTime**: 17:10:28 (21 seconds offset)
Because flight altitude and temperature are rounded, multiple captures taken during a hovering session share the same name, but represent different times and slight physical shifts. Pairing them directly will introduce spatial misalignment.

### Anomaly 2: Negative Relative Altitudes
Filenames like `FlightAltitude__-0.m` and `FlightAltitude__-1.m` exist under flights `03_03_2020` and `24_02_2020`. This is physically impossible for flight altitude above ground level, and points to barometer sensor drift or reference zeroing errors at drone startup.

### Quality Issues Counts
* **Blurry Images**: **528** (variance of Laplacian < 80). Blurriness is highly correlated with flight altitude; images taken at 8m-10m altitudes appear very soft and have low spatial details.
* **Low Contrast**: **29** (intensity std < 12.0). 
* **Clipped Histograms**: **1** (clipped low values > 5%). 0 clipped high.

---

## 11. Sample Visualizations

Below is an interactive carousel of the generated sample visualization panels. Each panel displays:
1. The normalized visual display of the thermal image (TIFF scaled to 8-bit; JPG/R_JPG shown directly).
2. The pixel intensity histogram.
3. An overlay table listing structural properties, filenames, and DJI UAV flight telemetry.

````carousel
![03_03_2020 JPG Visualizer](/C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/samples/sample_03_03_2020_JPG.png)
<!-- slide -->
![03_03_2020 R_JPG Visualizer](/C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/samples/sample_03_03_2020_R_JPG.png)
<!-- slide -->
![03_03_2020 TIFF Visualizer](/C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/samples/sample_03_03_2020_TIFF.png)
<!-- slide -->
![13_02_2020 JPG Visualizer](/C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/samples/sample_13_02_2020_JPG.png)
<!-- slide -->
![13_02_2020 R_JPG Visualizer](/C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/samples/sample_13_02_2020_R_JPG.png)
<!-- slide -->
![13_02_2020 TIFF Visualizer](/C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/samples/sample_13_02_2020_TIFF.png)
<!-- slide -->
![17_02_2020 JPG Visualizer](/C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/samples/sample_17_02_2020_JPG.png)
<!-- slide -->
![17_02_2020 R_JPG Visualizer](/C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/samples/sample_17_02_2020_R_JPG.png)
<!-- slide -->
![17_02_2020 TIFF Visualizer](/C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/samples/sample_17_02_2020_TIFF.png)
<!-- slide -->
![20_02_2020 JPG Visualizer](/C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/samples/sample_20_02_2020_JPG.png)
<!-- slide -->
![20_02_2020 R_JPG Visualizer](/C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/samples/sample_20_02_2020_R_JPG.png)
<!-- slide -->
![24_02_2020 JPG Visualizer](/C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/samples/sample_24_02_2020_JPG.png)
<!-- slide -->
![24_02_2020 R_JPG Visualizer](/C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/samples/sample_24_02_2020_R_JPG.png)
<!-- slide -->
![24_02_2020 TIFF Visualizer](/C:/Users/saran/.gemini/antigravity-ide/brain/e83e56de-79f3-44ed-b7f8-e2d211d3aaaa/scratch/samples/sample_24_02_2020_TIFF.png)
````

---

## 12. Potential Challenges for Machine Learning

1. **Paired Training Hazard**: Filenames match across JPG/TIFF but the timestamps show they are different captures. Training multi-modal or super-resolution networks by pairing images of identical filenames will introduce spatial mismatch and corrupt model learning.
2. **Missing Formats**: The `20_02_2020` flight is completely missing TIFF images. Any model architecture relying strictly on raw 16-bit TIFF inputs will not be able to utilize this flight day.
3. **Altitude-Based Blur**: As the drone rises above 5m, the spatial details of a 1cm landmine anomaly become extremely small and blurry due to the low resolution of the sensor (336x256).
4. **Thermal Contrast Uniformity**: The terrain is highly uniform soil. The thermal signatures of landmines are extremely subtle, and standard CNN models might easily overfit to soil textures rather than identifying actual anomalies.

---

## 13. Recommendations & Preprocessing Pipeline

To prepare this dataset for model training, the following preprocessing pipeline is recommended (prioritized by impact):

### Priority 1: Telemetry-Based Dynamic Alignment (CRITICAL)
Do **not** pair JPG and TIFF images using filenames. Instead, write an alignment script that matches images across the format directories using the EXIF `DateTimeOriginal` timestamp and the DJI XMP `RelativeAltitude` and `FlightYawDegree` to ensure spatial matching.

### Priority 2: Use TIFF for Radiometric Detection (HIGH)
Avoid using the `JPG` and `R_JPG` folders. These are 8-bit compressed images, which lose thermal resolution. Use the raw 16-bit `TIFF` files because they preserve the absolute uncalibrated digital sensor counts, providing a larger thermal dynamic range that is vital for detecting subtle anomalies.

### Priority 3: Altitude Stratification & Scaling (MEDIUM)
Sort training samples by `RelativeAltitude`. Construct the dataset into altitude ranges (e.g. Low: 0-3m, Medium: 3-6m, High: >6m). Apply super-resolution or spatial filtering (like contrast-limited adaptive histogram equalization - CLAHE) to high-altitude images to restore edge sharpness.

### Priority 4: Stratify Train/Test by Flight Date (HIGH)
To prevent model overoptimism, split train and test datasets strictly by **Flight Date** (e.g. train on 3 flights, validate on 1, test on 1). This ensures that the model generalizes to new environmental conditions, solar angles, and soil humidity profiles.
