from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class RiceDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.transform = transform

        # class folders automatically map
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            class_dir = self.root / cls
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((img_path, self.class_to_idx[cls]))

        print(f"[RiceDataset] Found {len(self.samples)} images across {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
