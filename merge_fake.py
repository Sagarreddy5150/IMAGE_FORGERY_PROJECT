import os
import shutil

source_root = r"C:\Users\sagar\Downloads\archive (1)\FF++C32-Frames"
print("Folders:", os.listdir(source_root))
target_fake = r"C:\Users\sagar\OneDrive\Pictures\Documents\image_forgery_project\dataset\fake"

fake_folders = [
    "Deepfakes",
    "FaceSwap",
    "Face2Face",
    "FaceShifter",
    "NeuralTextures"
]

os.makedirs(target_fake, exist_ok=True)

count = 0
print("Available folders:", os.listdir(source_root))
for folder in fake_folders:
    folder_path = os.path.join(source_root, folder)

    if not os.path.exists(folder_path):
        print("❌ Folder not found:", folder_path)
        continue

    print(f"Processing {folder}...")

    for i, file in enumerate(os.listdir(folder_path)):
        src = os.path.join(folder_path, file)
        dst = os.path.join(target_fake, f"{folder}_{file}")

        shutil.copy(src, dst)
        count += 1

        if i % 500 == 0:
            print(f"{folder}: {i} files copied...")

print("✅ DONE. Total files:", count)