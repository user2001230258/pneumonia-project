# ============================================================
# dataset.py
# ============================================================
import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

IMG_SIZE    = 224
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
BATCH_SIZE  = 32
NUM_WORKERS = 4
RANDOM_SEED = 42

SPLIT_DIR   = "/kaggle/working"  

LABEL2IDX = {"NORMAL": 0, "PNEUMONIA": 1}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}



def get_transforms(split: str) -> T.Compose:

    base = [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ]

    if split == "train":
        aug = [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
        return T.Compose(aug)

    return T.Compose(base)   


class ChestXrayDataset(Dataset):


    def __init__(self, df: pd.DataFrame, transform=None, split: str = "train"):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.split     = split

        missing = self.df[~self.df["image_path"].apply(os.path.exists)]
        if len(missing) > 0:
            print(f"  !  [{split}] {len(missing):,} ảnh không tồn tại — loại bỏ")
            self.df = self.df[self.df["image_path"].apply(os.path.exists)].reset_index(drop=True)

        self.labels = self.df["label"].map(LABEL2IDX).values

        counts = pd.Series(self.labels).value_counts().to_dict()
        print(f"  [{split:<5}] {len(self.df):>6,} ảnh  | "
              f"NORMAL={counts.get(0,0):>5,}  PNEUMONIA={counts.get(1,0):>5,}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

    def get_class_weights(self) -> torch.Tensor:
 
        counts  = np.bincount(self.labels)
        weights = len(self.labels) / (len(counts) * counts)
        return torch.tensor(weights, dtype=torch.float32)



def build_dataloaders(
    split_dir: str = SPLIT_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple:
   
    print("\n" + "=" * 65)
    print("       TẠO PYTORCH DATASET & DATALOADER")
    print("=" * 65)

    splits  = {}
    loaders = {}

    for split in ["train", "val", "test"]:
        csv_path = os.path.join(split_dir, f"split_{split}.csv")
        df       = pd.read_csv(csv_path)
        transform = get_transforms(split)
        dataset   = ChestXrayDataset(df, transform=transform, split=split)

        shuffle    = (split == "train")
        drop_last  = (split == "train")
        pin_memory = torch.cuda.is_available()

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        splits[split] = dataset

    class_weights = splits["train"].get_class_weights()

    print(f"\n  Batch size   : {batch_size}")
    print(f"  Num workers  : {num_workers}")
    print(f"  IMG size     : {IMG_SIZE}×{IMG_SIZE}")
    print(f"\n  Class weights (dùng cho WeightedLoss):")
    for idx, lbl in IDX2LABEL.items():
        print(f"    {lbl:<12}: {class_weights[idx]:.4f}")

    print(f"\n  Số batch/epoch:")
    for split, loader in loaders.items():
        print(f"    {split:<5}: {len(loader):>5} batches")

    print("=" * 65)

    return (
        loaders["train"],
        loaders["val"],
        loaders["test"],
        class_weights,
    )


def sanity_check(loader: DataLoader, split: str = "train"):

    images, labels = next(iter(loader))
    print(f"\n  [{split}] Sanity check:")
    print(f"    images.shape : {images.shape}   dtype={images.dtype}")
    print(f"    labels.shape : {labels.shape}   dtype={labels.dtype}")
    print(f"    pixel mean   : {images.mean():.4f}   std={images.std():.4f}")

    unique, counts = labels.unique(return_counts=True)
    for u, c in zip(unique.tolist(), counts.tolist()):
        print(f"    label {u} ({IDX2LABEL[u]:<12}): {c} samples trong batch")

def get_dataloaders(
    split_dir: str  = SPLIT_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple:

    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        split_dir, batch_size, num_workers
    )
    sanity_check(train_loader, "train")
    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_weights = get_dataloaders()


#==========================================================
#ADD DATASETS: 'nih-chest-xray-data-preprocessing-output'
#==========================================================
