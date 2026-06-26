import pandas as pd

df = pd.read_csv(r"D:\DroneML\results\metadata_checked.csv")

print("="*60)
print("DATASET HEALTH REPORT")
print("="*60)

print(f"Total Images            : {len(df)}")
print(f"Mine Images             : {(df['label']=='Mine').sum()}")
print(f"No Mine Images          : {(df['label']=='No_Mine').sum()}")

print("\nFlights")
print(df["flight_date"].value_counts())

print("\nFormats")
print(df["format_folder"].value_counts())

print("\nDepths")
print(df["depth"].value_counts())

print("\nImage Extensions")
print(df["extension"].value_counts())

print("\nImage Formats")
print(df["actual_format"].value_counts())

print("\nImage Modes")
print(df["mode"].value_counts())

print("\nImage Sizes")
print(df.groupby(["width","height"]).size())

print("\nFolder Extension Mismatch")
print((~df["folder_extension_match"]).sum())

print("\nCorrupted Images")
print(df["corrupted"].sum())

print("="*60)

df.groupby("flight_date").size().to_csv(
    r"D:\DroneML\results\flight_distribution.csv"
)

df.groupby("depth").size().to_csv(
    r"D:\DroneML\results\depth_distribution.csv"
)

df.groupby("label").size().to_csv(
    r"D:\DroneML\results\class_distribution.csv"
)

print("CSV Reports Saved.")