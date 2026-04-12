# ============================================================
# data_preprocessing.py
# ============================================================
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from sklearn.model_selection import train_test_split

IMG_SIZE   = 224          
MEAN       = [0.485, 0.456, 0.406]  
STD        = [0.229, 0.224, 0.225] 

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

FILTERED_CSV = "/kaggle/working/filtered_dataset_2class.csv"
OUTPUT_DIR   = "/kaggle/working"


def show_preprocessing_samples(df: pd.DataFrame, n_per_class: int = 3):

    print( "-" * 65)
    print("       MINH HỌA TRƯỚC / SAU PREPROCESSING")
    print("=" * 65)

    labels    = ["NORMAL", "PNEUMONIA"]
    n_classes = len(labels)
    fig       = plt.figure(figsize=(n_per_class * 4, n_classes * 4))
    gs        = gridspec.GridSpec(n_classes * 2, n_per_class, hspace=0.5, wspace=0.3)

    for row_idx, lbl in enumerate(labels):
        sub     = df[df["label"] == lbl].sample(n=n_per_class, random_state=RANDOM_SEED)
        samples = sub["image_path"].tolist()

        for col_idx, img_path in enumerate(samples):
            img_orig = Image.open(img_path).convert("RGB")

            ax_orig = fig.add_subplot(gs[row_idx * 2, col_idx])
            ax_orig.imshow(img_orig, cmap="gray")
            ax_orig.set_title(f"[{lbl}] Gốc\n{img_orig.size}", fontsize=8)
            ax_orig.axis("off")

            img_resized = img_orig.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            arr         = np.array(img_resized).astype(np.float32) / 255.0
            arr_norm    = (arr - np.array(MEAN)) / np.array(STD)

            arr_disp    = np.clip((arr_norm - arr_norm.min()) /
                                  (arr_norm.max() - arr_norm.min() + 1e-8), 0, 1)

            ax_proc = fig.add_subplot(gs[row_idx * 2 + 1, col_idx])
            ax_proc.imshow(arr_disp)
            ax_proc.set_title(f"Sau resize+norm\n({IMG_SIZE}×{IMG_SIZE})", fontsize=8)
            ax_proc.axis("off")

    plt.suptitle("Minh hoạ Preprocessing – Trước / Sau", fontsize=13, fontweight="bold")
    out = os.path.join(OUTPUT_DIR, "preprocessing_samples.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print("=" * 65)



def split_dataset(df: pd.DataFrame) -> tuple:
    print("       CHIA TRAIN / VAL / TEST")
    print("=" * 65)

    df_train, df_temp = train_test_split(
        df,
        test_size=1 - TRAIN_RATIO,
        random_state=RANDOM_SEED,
        stratify=df["label"]
    )

    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=1 - val_ratio_adjusted,
        random_state=RANDOM_SEED,
        stratify=df_temp["label"]
    )

    for name, subset in [("TRAIN", df_train), ("VAL  ", df_val), ("TEST ", df_test)]:
        counts = subset["label"].value_counts()
        print(f"  {name} : {len(subset):>6,} ảnh  | "
              f"NORMAL={counts.get('NORMAL',0):>5,}  "
              f"PNEUMONIA={counts.get('PNEUMONIA',0):>5,}")

    print(f"\n  Tỉ lệ  : Train={TRAIN_RATIO:.0%}  Val={VAL_RATIO:.0%}  Test={TEST_RATIO:.0%}")


    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        out_path = os.path.join(OUTPUT_DIR, f"split_{name}.csv")
        subset.to_csv(out_path, index=False)

    print("=" * 65)
    return df_train, df_val, df_test



def show_augmentation_preview(df: pd.DataFrame, n_samples: int = 4):

    import torchvision.transforms as T
    import torch

    print("       MINH HỌA AUGMENTATION")
    print("=" * 65)

    augmentations = {
        "Gốc (resize)": T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
        ]),
        "Horizontal Flip": T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(p=1.0),
        ]),
        "Rotation ±15°": T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomRotation(15),
        ]),
        "ColorJitter": T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ColorJitter(brightness=0.3, contrast=0.3),
        ]),
    }

    samples = df.sample(n=n_samples, random_state=RANDOM_SEED)["image_path"].tolist()
    n_aug   = len(augmentations)

    fig, axes = plt.subplots(n_samples, n_aug, figsize=(n_aug * 3, n_samples * 3))

    for row, img_path in enumerate(samples):
        img_pil = Image.open(img_path).convert("RGB")
        for col, (aug_name, transform) in enumerate(augmentations.items()):
            aug_img = transform(img_pil)
            axes[row, col].imshow(aug_img)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(aug_name, fontsize=9, fontweight="bold")

    plt.suptitle("Minh hoạ Augmentation – Train set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "augmentation_preview.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print("=" * 65)


def run_preprocessing() -> tuple:
    df = pd.read_csv(FILTERED_CSV)
    print(f"  Đã load: {len(df):,} ảnh\n")


    show_preprocessing_samples(df)
    df_train, df_val, df_test = split_dataset(df)

    try:
        show_augmentation_preview(df_train)
    except ImportError:
        print("    torchvision chưa cài — bỏ qua augmentation preview")

    print("   PREPROCESSING HOÀN TẤT!")
    print(f"  → split_train.csv / split_val.csv / split_test.csv")
    print("=" * 65)

    return df_train, df_val, df_test


if __name__ == "__main__":
    df_train, df_val, df_test = run_preprocessing()
