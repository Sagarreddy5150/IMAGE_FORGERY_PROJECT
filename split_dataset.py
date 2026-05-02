import os
import shutil
import random

source = "dataset"
train_dir = "dataset/train"
val_dir = "dataset/val"

for category in ["real", "fake"]:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

    files = os.listdir(os.path.join(source, category))
    random.shuffle(files)

    split = int(0.8 * len(files))

    train_files = files[:split]
    val_files = files[split:]

    for f in train_files:
        shutil.copy(os.path.join(source, category, f),
                    os.path.join(train_dir, category, f))

    for f in val_files:
        shutil.copy(os.path.join(source, category, f),
                    os.path.join(val_dir, category, f))