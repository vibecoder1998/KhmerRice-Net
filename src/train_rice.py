import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

from src.rice_dataset import RiceDiseaseDataset
from src.model_def import RiceCNN


def build_loaders(data_dir, batch_size=32, img_size=224):

    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataset_full = RiceDiseaseDataset(data_dir, transform_train)

    indices = list(range(len(dataset_full)))
    labels = [lbl for _, lbl in dataset_full.samples]

    train_idx, val_idx = train_test_split(indices, test_size=0.2,
                                          stratify=labels, random_state=42)

    train_subset = torch.utils.data.Subset(dataset_full, train_idx)
    val_subset = torch.utils.data.Subset(
        RiceDiseaseDataset(data_dir, transform_val), val_idx
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    return train_loader, val_loader, dataset_full.classes


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/rice")
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, classes = build_loaders(args.data)
    num_classes = len(classes)

    model = RiceCNN(backbone=args.backbone, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Stage 1 — train only classifier
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    print("\n===== Stage 1: Training classifier only =====")
    for epoch in range(1, 4):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        print(f"[Head] Epoch {epoch} Train Acc={train_acc:.3f} Val Acc={val_acc:.3f}")

    # Stage 2 — unfreeze backbone
    print("\n===== Stage 2: Fine-tuning entire model =====")
    model.unfreeze_backbone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0
    for epoch in range(4, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        print(f"[Full] Epoch {epoch} Train Acc={train_acc:.3f} Val Acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "backbone": args.backbone
            }, "src/models/rice_cnn_model.pth")
            print(f"Saved best model with acc={best_acc:.3f}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)

        loss_sum += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, correct / total


if __name__ == "__main__":
    train()
