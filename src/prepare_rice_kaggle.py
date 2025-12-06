import shutil
from pathlib import Path
import kagglehub

print("Downloading rice dataset from KaggleHub...")
base_path = Path(kagglehub.dataset_download("anshulm257/rice-disease-dataset"))
print("Dataset downloaded to:", base_path)

possible_dirs = [d for d in base_path.iterdir() if d.is_dir()]
if len(possible_dirs) == 1:
    src_root = possible_dirs[0]
else:
    # fallback: find folder containing rice classes
    for d in possible_dirs:
        if "rice" in d.name.lower() or "leaf" in d.name.lower():
            src_root = d
            break
    else:
        raise RuntimeError("Could not locate main dataset folder inside KaggleHub download.")

print("Root dataset folder detected:", src_root)

# Where dataset should be placed
target_root = Path("src/data/rice")
target_root.mkdir(parents=True, exist_ok=True)

# Mapping REAL folder names to your internal class labels
FOLDER_MAP = {
    "Bacterial Leaf Blight": "bacterial_leaf_blight",
    "Brown Spot": "brown_spot",
    "Healthy Rice Leaf": "healthy",
    "Leaf Blast": "leaf_blast",
    "Leaf scald": "leaf_scald",
    "Sheath Blight": "sheath_blight",
}

print("\n Preparing dataset folders...\n")

for src_name, dst_name in FOLDER_MAP.items():

    src_folder = src_root / src_name
    dst_folder = target_root / dst_name

    print(f"Checking: {src_folder}")

    if not src_folder.exists():
        print(f"Warning: Folder '{src_name}' not found in dataset.")
        continue

    dst_folder.mkdir(parents=True, exist_ok=True)

    for img in src_folder.glob("*"):
        if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            shutil.copy(img, dst_folder)

    print(f"Copied: {src_name} → {dst_name}")

print("\nDataset ready!")
print("Final dataset structure:")
for d in target_root.iterdir():
    print(" -", d)
