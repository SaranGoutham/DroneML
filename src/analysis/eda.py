import os
import pandas as pd
import matplotlib.pyplot as plt

# =====================================
# CONFIGURATION
# =====================================

METADATA_FILE = r"D:\DroneML\results\metadata_checked.csv"

OUTPUT_DIR = r"D:\DroneML\results\graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(METADATA_FILE)

print("="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

print(f"Total Images : {len(df)}")

# =====================================
# CLASS DISTRIBUTION
# =====================================

plt.figure(figsize=(6,5))
df["label"].value_counts().plot(kind="bar")
plt.title("Mine vs No Mine")
plt.xlabel("Class")
plt.ylabel("Images")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"class_distribution.png"))
plt.close()

# =====================================
# FLIGHT DISTRIBUTION
# =====================================

plt.figure(figsize=(7,5))
df["flight_date"].value_counts().sort_index().plot(kind="bar")
plt.title("Flight Distribution")
plt.xlabel("Flight Date")
plt.ylabel("Images")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"flight_distribution.png"))
plt.close()

# =====================================
# FORMAT DISTRIBUTION
# =====================================

plt.figure(figsize=(6,5))
df["format_folder"].value_counts().plot(kind="bar")
plt.title("Image Format Distribution")
plt.xlabel("Format")
plt.ylabel("Images")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"format_distribution.png"))
plt.close()

# =====================================
# DEPTH DISTRIBUTION
# =====================================

plt.figure(figsize=(7,5))
df["depth"].value_counts().plot(kind="bar")
plt.title("Mine Depth Distribution")
plt.xlabel("Depth")
plt.ylabel("Images")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"depth_distribution.png"))
plt.close()

# =====================================
# IMAGE FORMAT
# =====================================

plt.figure(figsize=(6,5))
df["actual_format"].value_counts().plot(kind="bar")
plt.title("Actual Image Format")
plt.xlabel("Format")
plt.ylabel("Images")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"actual_format_distribution.png"))
plt.close()

# =====================================
# IMAGE MODE
# =====================================

plt.figure(figsize=(6,5))
df["mode"].value_counts().plot(kind="bar")
plt.title("Image Mode")
plt.xlabel("Mode")
plt.ylabel("Images")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"mode_distribution.png"))
plt.close()

# =====================================
# RESOLUTION
# =====================================

resolution = (
    df["width"].astype(str)
    + "x"
    + df["height"].astype(str)
)

plt.figure(figsize=(6,5))
resolution.value_counts().plot(kind="bar")
plt.title("Image Resolution")
plt.xlabel("Resolution")
plt.ylabel("Images")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"resolution_distribution.png"))
plt.close()

# =====================================
# ZONE DISTRIBUTION
# =====================================

plt.figure(figsize=(12,5))
df["zone"].value_counts().plot(kind="bar")
plt.title("Zone Distribution")
plt.xlabel("Zone")
plt.ylabel("Images")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"zone_distribution.png"))
plt.close()

# =====================================
# FILE SIZE
# =====================================

plt.figure(figsize=(7,5))
df["file_size_kb"].hist(bins=30)
plt.title("File Size Distribution")
plt.xlabel("KB")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"file_size_distribution.png"))
plt.close()

# =====================================
# SUMMARY
# =====================================

summary = pd.DataFrame({
    "Metric":[
        "Total Images",
        "Mine Images",
        "No Mine Images",
        "Flights",
        "Formats",
        "Zones",
        "Corrupted Images",
        "Extension Mismatch"
    ],
    "Value":[
        len(df),
        (df["label"]=="Mine").sum(),
        (df["label"]=="No_Mine").sum(),
        df["flight_date"].nunique(),
        df["format_folder"].nunique(),
        df["zone"].nunique(),
        df["corrupted"].sum(),
        (~df["folder_extension_match"]).sum()
    ]
})

summary.to_csv(
    os.path.join(OUTPUT_DIR,"dataset_summary.csv"),
    index=False
)

print("\nGraphs Saved To:")
print(OUTPUT_DIR)

print("\nGenerated Files")

for f in sorted(os.listdir(OUTPUT_DIR)):
    print("✓",f)

print("\nEDA COMPLETED SUCCESSFULLY")