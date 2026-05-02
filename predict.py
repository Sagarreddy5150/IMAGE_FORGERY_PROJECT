import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def parse_args():
    parser = argparse.ArgumentParser(description="Run image forgery detection inference.")
    parser.add_argument("path", help="Path to an image file or folder of images.")
    parser.add_argument("--model", default="best_model.keras", help="Path to the saved model file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for fake detection.")
    return parser.parse_args()


def load_forgery_model(path):
    if not os.path.exists(path):
        print(f"❌ Model file '{path}' not found.")
        sys.exit(1)
    model = load_model(path)
    print(f"✅ Model loaded from {path}")
    return model

args = parse_args()
model = load_forgery_model(args.model)

COLOR_FAKE = (0, 0, 255)
COLOR_REAL = (0, 200, 0)
COLOR_BG = (20, 20, 20)
FONT = cv2.FONT_HERSHEY_SIMPLEX
THRESHOLD = args.threshold

def draw_result_on_image(img_display, is_fake, confidence):
    h, w = img_display.shape[:2]
    color = COLOR_FAKE if is_fake else COLOR_REAL

    # Top banner
    overlay = img_display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.80, img_display, 0.20, 0, img_display)

    cv2.putText(img_display, "IMAGE FORGERY DETECTION SYSTEM",
                (10, 22), FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    result_text = "FAKE IMAGE" if is_fake else "REAL IMAGE"
    cv2.putText(img_display, result_text,
                (10, 65), FONT, 1.5, color, 3, cv2.LINE_AA)

    # Bottom confidence bar
    bar_y = h - 60
    cv2.rectangle(img_display, (0, bar_y), (w, h), COLOR_BG, -1)
    cv2.rectangle(img_display, (10, bar_y + 10), (w - 10, bar_y + 35), (60, 60, 60), -1)
    bar_fill = int((w - 20) * confidence)
    cv2.rectangle(img_display, (10, bar_y + 10), (10 + bar_fill, bar_y + 35), color, -1)
    cv2.putText(img_display, f"Confidence: {confidence * 100:.1f}%",
                (10, h - 8), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Border
    cv2.rectangle(img_display, (2, 2), (w - 2, h - 2), color, 5)
    return img_display


def predict_and_show(img_path):
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return None

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"❌ Could not read: {img_path}")
        return None

    # Model input
    img_input = cv2.resize(img_bgr, (224, 224)) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    prediction = model.predict(img_input, verbose=0)[0][0]

    is_fake = prediction > THRESHOLD
    confidence = float(prediction) if is_fake else float(1 - prediction)

    # Terminal output
    icon = "🔴" if is_fake else "🟢"
    label = "FAKE" if is_fake else "REAL"
    print(f"\n{icon} {label}  |  Confidence: {confidence*100:.2f}%  |  {os.path.basename(img_path)}")

    # Display image
    disp = cv2.resize(img_bgr, (700, 520))
    disp = draw_result_on_image(disp, is_fake, confidence)
    cv2.imshow("Image Forgery Detection  [Press any key = next | Q = quit]", disp)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    return prediction, key


def predict_folder_visual(folder_path):
    if not os.path.isdir(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(exts)])

    if not images:
        print("❌ No images found.")
        return

    print(f"\n📁 {len(images)} images found in: {folder_path}")
    print("Controls: Any key = next image | Q = quit\n")

    fake_count = real_count = 0

    for img_file in images:
        result = predict_and_show(os.path.join(folder_path, img_file))
        if result is None:
            continue
        pred, key = result
        if pred > 0.6:
            fake_count += 1
        else: 
            real_count += 1
        if key in [ord('q'), ord('Q')]:
            break

    print(f"\n{'='*40}")
    print(f"🟢 Real: {real_count}  |  🔴 Fake: {fake_count}  |  Total: {real_count + fake_count}")


# ─── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    target_path = args.path
    if os.path.isdir(target_path):
        predict_folder_visual(target_path)
    else:
        predict_and_show(target_path)
