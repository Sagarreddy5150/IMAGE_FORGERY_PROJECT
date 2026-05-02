# Image Forgery Detection Project

A deep learning system to detect **fake (forged/manipulated) images** from real ones using **MobileNetV2 transfer learning** with TensorFlow/Keras.

---

## 📁 Project Structure

```text
image_forgery_project/
├── dataset/
│   ├── train/
│   │   ├── fake/
│   │   └── real/
│   └── val/
│       ├── fake/
│       └── real/
├── merge_fake.py         # Merges fake image folders from FF++ dataset
├── merge_datasets.py     # Dataset organization / preprocessing
├── split_dataset.py      # Splits dataset into train/validation sets
├── train_model.py        # Trains the MobileNetV2 model
├── predict.py            # Predicts real/fake on new images
├── README.md
└── .gitignore
```

---

## 📦 Dataset

* **Source**: FaceForensics++ (FF++) dataset — C23 compression frames
* **Fake categories**:

  * Deepfakes
  * FaceSwap
  * Face2Face
  * FaceShifter
  * NeuralTextures
* **Real category**:

  * Original YouTube face videos

---

## 🧠 Model Architecture

* **Base Model**: MobileNetV2 (pretrained on ImageNet)
* **Transfer Learning** used for feature extraction
* **Fine-tuning** applied to higher layers
* **Classification Head**:

  * GlobalAveragePooling2D
  * BatchNormalization
  * Dense Layer (ReLU)
  * Dropout
  * Dense(1, Sigmoid)

---

## ⚙️ How to Run

### 1. Install Requirements

```bash
pip install tensorflow opencv-python matplotlib scikit-learn numpy
```

### 2. Prepare Dataset

```bash
python merge_fake.py
python split_dataset.py
```

### 3. Train the Model

```bash
python train_model.py
```

**Output:**

* Trained model file saved locally
* Accuracy/Loss graphs generated

### 4. Predict on New Images

```bash
# Single image
python predict.py test.jpg

# Entire folder
python predict.py path/to/folder/
```

---

## 📊 Training Details

| Parameter         | Value                       |
| ----------------- | --------------------------- |
| Image Size        | 224 × 224                   |
| Batch Size        | 32                          |
| Max Epochs        | 30                          |
| Optimizer         | Adam                        |
| Loss Function     | Binary Crossentropy         |
| Data Augmentation | Rotation, Zoom, Flip, Shift |
| Class Balancing   | Weighted Loss               |

---

## 🔍 Current Performance

* Predicts whether an image is **Real** or **Fake**
* Displays confidence score during prediction
* Supports real-time image testing
* Transfer learning improves performance on limited datasets

---

## 🛠️ Technologies Used

* Python 3.x
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* scikit-learn

---

## 🚀 Future Improvements

* Improve accuracy using EfficientNet / Xception
* Add video forgery detection
* Build web deployment using Flask / Streamlit
* Add explainable AI heatmaps

---

## 👨‍💻 Author

**Dalli Sagar Durga Pradeep**
