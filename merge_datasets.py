"""
Add FaceForensics++ dataset directly to existing dataset folder.
Balances classes by sampling and maintains 80/20 train/val split.
"""
import os
import shutil
import random

# Paths
dataset = r"dataset"
ff_dataset = r"C:\Users\sagar\Downloads\archive (1)\FF++C32-Frames"

def add_faceforensics():
    """Add FaceForensics++ data directly to dataset folder."""
    print("Adding FaceForensics++ to dataset...")
    
    # Collect all Original (real) images
    real_images = []
    ff_real_dir = os.path.join(ff_dataset, "Original")
    if os.path.isdir(ff_real_dir):
        for fname in os.listdir(ff_real_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                real_images.append(os.path.join(ff_real_dir, fname))
    
    # Collect all fake images (all manipulation types)
    fake_images = []
    for manip_type in ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]:
        manip_dir = os.path.join(ff_dataset, manip_type)
        if os.path.isdir(manip_dir):
            for fname in os.listdir(manip_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    fake_images.append(os.path.join(manip_dir, fname))
    
    print(f"Found {len(real_images)} real (Original) images")
    print(f"Found {len(fake_images)} fake (manipulated) images")
    
    # Balance: use equivalent fake count to real count
    total_real = len(real_images)
    fake_to_use = min(len(fake_images), total_real)
    sampled_fake = random.sample(fake_images, fake_to_use)
    
    print(f"Using {total_real} real + {fake_to_use} fake (balanced)")
    
    # Split 80/20
    train_split = 0.8
    train_real_count = int(total_real * train_split)
    train_fake_count = int(fake_to_use * train_split)
    
    random.shuffle(real_images)
    random.shuffle(sampled_fake)
    
    # Add real images
    print("\nAdding real images...")
    for i, src_file in enumerate(real_images):
        split = "train" if i < train_real_count else "val"
        fname = os.path.basename(src_file)
        dst_file = os.path.join(dataset, split, "real", f"ff_{i:04d}_{fname}")
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy2(src_file, dst_file)
        if (i + 1) % 1000 == 0:
            print(f"  Copied {i + 1}/{total_real} real images")
    
    # Add fake images
    print("Adding fake images...")
    for i, src_file in enumerate(sampled_fake):
        split = "train" if i < train_fake_count else "val"
        fname = os.path.basename(src_file)
        dst_file = os.path.join(dataset, split, "fake", f"ff_{i:04d}_{fname}")
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy2(src_file, dst_file)
        if (i + 1) % 1000 == 0:
            print(f"  Copied {i + 1}/{fake_to_use} fake images")
    
    print_summary()

def print_summary():
    """Print final dataset statistics."""
    print("\n" + "="*60)
    print("UPDATED DATASET SUMMARY")
    print("="*60)
    
    for split in ["train", "val"]:
        print(f"\n{split.upper()}:")
        real_dir = os.path.join(dataset, split, "real")
        fake_dir = os.path.join(dataset, split, "fake")
        real_count = len([f for f in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, f))])
        fake_count = len([f for f in os.listdir(fake_dir) if os.path.isfile(os.path.join(fake_dir, f))])
        print(f"  Real: {real_count}")
        print(f"  Fake: {fake_count}")
        total = real_count + fake_count
        ratio = fake_count / real_count if real_count > 0 else 0
        print(f"  Total: {total} | Fake/Real ratio: {ratio:.2f}")
    
    print("\n" + "="*60)
    print("Ready to train!")
    print("Run: python train_model.py --epochs 30 --batch-size 16")
    print("="*60)

def main():
    random.seed(42)
    add_faceforensics()

if __name__ == "__main__":
    main()
