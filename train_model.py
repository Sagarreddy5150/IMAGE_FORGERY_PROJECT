import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight


def parse_args():
    parser = argparse.ArgumentParser(description="Train a forgery detection model using transfer learning.")
    parser.add_argument("--data-dir", default="dataset", help="Root dataset folder containing train/ and val/ directories.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image width and height.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--model-path", default="best_model.keras", help="Path to save the best model.")
    parser.add_argument("--plot-path", default="training_history.png", help="Path to save training history plot.")
    return parser.parse_args()


def build_datasets(data_dir, image_size, batch_size):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_ds, class_names, train_counts = build_balanced_train_dataset(train_dir, image_size, batch_size)
    val_ds = image_dataset_from_directory(
        val_dir,
        label_mode="binary",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False
    )

    print(f"Class names: {class_names}")
    print(f"Training class counts before balancing: {train_counts}")
    print("Training dataset balanced for equal class representation.")

    return train_ds, val_ds, class_names


def get_class_weights(dataset):
    labels = []
    for _, y_batch in dataset:
        labels.append(y_batch.numpy().reshape(-1))
    labels = np.concatenate(labels)
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    return dict(enumerate(weights))


def load_and_preprocess_image(path, label, image_size):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [image_size, image_size])
    return image, label


def build_balanced_train_dataset(train_dir, image_size, batch_size, seed=42):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    paths = []
    labels = []
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(train_dir, class_name)
        for fname in os.listdir(class_path):
            if fname.lower().endswith(valid_extensions):
                paths.append(os.path.join(class_path, fname))
                labels.append(idx)

    paths = np.array(paths)
    labels = np.array(labels)

    train_counts = {name: int(np.sum(labels == idx)) for idx, name in enumerate(class_names)}
    max_count = max(train_counts.values())

    np.random.seed(seed)
    balanced_indices = []
    for idx in range(len(class_names)):
        idxs = np.where(labels == idx)[0]
        if len(idxs) < max_count:
            sampled = np.random.choice(idxs, size=max_count, replace=True)
        else:
            sampled = np.random.choice(idxs, size=max_count, replace=False)
        balanced_indices.append(sampled)

    balanced_indices = np.concatenate(balanced_indices)
    np.random.shuffle(balanced_indices)

    balanced_paths = paths[balanced_indices]
    balanced_labels = labels[balanced_indices]

    dataset = tf.data.Dataset.from_tensor_slices((balanced_paths, balanced_labels))
    dataset = dataset.shuffle(len(balanced_paths), seed=seed)
    dataset = dataset.map(lambda path, label: load_and_preprocess_image(path, label, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, class_names, train_counts


def create_model(image_size):
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.05),
    ], name="augmentation")

    inputs = layers.Input(shape=(image_size, image_size, 3), name="input_image")
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(0.4, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="forgery_detector")
    return model, base_model


def prepare_dataset(dataset, cache=False):
    AUTOTUNE = tf.data.AUTOTUNE
    if cache:
        dataset = dataset.cache()
    return dataset.prefetch(buffer_size=AUTOTUNE)


def plot_history(history, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.plot(history.history["auc"], label="train_auc")
    plt.plot(history.history["val_auc"], label="val_auc")
    plt.title("Training Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training plot to: {save_path}")


def plot_roc_pr_curves(y_true, y_scores, save_dir):
    roc_path = os.path.join(save_dir, "roc_curve.png")
    pr_path = os.path.join(save_dir, "precision_recall_curve.png")

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curve to: {roc_path}")

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="purple", lw=2, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()
    print(f"Saved Precision-Recall curve to: {pr_path}")


def evaluate_model(model, dataset, class_names, save_dir):
    y_true = []
    y_pred = []
    y_scores = []
    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0).reshape(-1)
        y_true.append(y_batch.numpy())
        y_scores.append(preds)
        y_pred.append((preds > 0.5).astype(int))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_scores = np.concatenate(y_scores)

    print("\nValidation classification report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_true, y_scores):.4f}")
    print(f"Average precision (AP): {average_precision_score(y_true, y_scores):.4f}")

    plot_roc_pr_curves(y_true, y_scores, save_dir)


def main():
    args = parse_args()

    tf.random.set_seed(42)
    np.random.seed(42)

    train_ds, val_ds, class_names = build_datasets(args.data_dir, args.image_size, args.batch_size)
    train_ds = prepare_dataset(train_ds, cache=False)
    val_ds = prepare_dataset(val_ds, cache=True)

    class_weights = get_class_weights(train_ds)
    print(f"Class weights: {class_weights}")

    model, base_model = create_model(args.image_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    callbacks = [
        EarlyStopping(monitor="val_auc", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(args.model_path, monitor="val_auc", save_best_only=True, save_weights_only=False, verbose=1),
    ]

    phase1_epochs = min(5, max(1, args.epochs // 2))
    fine_tune_epochs = max(1, args.epochs - phase1_epochs)
    total_epochs = phase1_epochs + fine_tune_epochs

    print(f"\nPhase 1: training top layers only for {phase1_epochs} epochs")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print(f"\nPhase 2: fine-tuning top of the pretrained network for {fine_tune_epochs} additional epochs")
    for layer in base_model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=phase1_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    plot_history(history2, args.plot_path)
    model.save(args.model_path)
    print(f"Best model saved to: {args.model_path}")

    plot_dir = os.path.dirname(args.plot_path) or "."
    os.makedirs(plot_dir, exist_ok=True)
    evaluate_model(model, val_ds, class_names, plot_dir)


if __name__ == "__main__":
    main()
