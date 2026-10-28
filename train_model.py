
import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

SEED = 42
IMG_SIZE = (200, 200)
BATCH_SIZE = 32
EPOCHS = 25

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Dataset path relative to this Python file
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "NEU-CLS"

IMAGE_DIRS = [
    DATASET_DIR / "train" / "train" / "images",
    DATASET_DIR / "valid" / "valid" / "images",
]

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. CHECK DATASET
# --------------------------------------------------

image_paths = []

for folder in IMAGE_DIRS:
    if folder.exists():
        image_paths.extend(
            p for p in folder.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        )
    else:
        print("Folder not found:", folder)

if not image_paths:
    raise FileNotFoundError(
        "No images found. Check your NEU-CLS folder structure."
    )

print("Total images found:", len(image_paths))

# --------------------------------------------------
# 3. EXTRACT CLASS NAMES FROM FILENAMES
# --------------------------------------------------

def get_class_name(path):
    # Example: crazing_1.jpg -> crazing
    # Example: rolled-in_scale_1.jpg -> rolled-in_scale
    return path.stem.rsplit("_", 1)[0]

class_names = sorted(set(get_class_name(p) for p in image_paths))
class_to_index = {
    name: index for index, name in enumerate(class_names)
}

print("\nClasses found:")
for name, index in class_to_index.items():
    print(index, ":", name)

if len(class_names) != 6:
    raise ValueError(
        f"Expected 6 classes, but found {len(class_names)}. "
        "Check your image filenames."
    )

# --------------------------------------------------
# 4. LOAD AND PREPARE IMAGES
# --------------------------------------------------

X = []
y = []
valid_paths = []

for i, path in enumerate(image_paths):
    try:
        with Image.open(path) as image:
            image = image.convert("L")
            image = image.resize(IMG_SIZE)
            X.append(np.array(image, dtype=np.uint8))

        y.append(class_to_index[get_class_name(path)])
        valid_paths.append(path)

    except Exception as error:
        print("Skipping unreadable image:", path, error)

X = np.array(X, dtype=np.float32)[..., np.newaxis] / 255.0
y = np.array(y, dtype=np.int32)

print("\nImage array shape:", X.shape)
print("Label array shape:", y.shape)

# Show the number of images in each class
counts = {
    name: sum(get_class_name(p) == name for p in valid_paths)
    for name in class_names
}

print("\nImages per class:")
for name, count in counts.items():
    print(f"{name}: {count}")

# --------------------------------------------------
# 5. SPLIT DATA: 80% TRAIN, 20% TEST
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))

# --------------------------------------------------
# 6. BUILD CNN MODEL
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(200, 200, 1)),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# 7. TRAIN MODEL
# --------------------------------------------------

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# --------------------------------------------------
# 8. PLOT ACCURACY AND LOSS
# --------------------------------------------------

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_graphs.png")
plt.show()

# --------------------------------------------------
# 9. EVALUATE ON HELD-OUT TEST SET
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

probabilities = model.predict(X_test, verbose=0)
predictions = np.argmax(probabilities, axis=1)

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=list(range(len(class_names)))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(xticks_rotation=45, values_format="d")
plt.title("Defect Classification Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.show()

# --------------------------------------------------
# 10. SAVE MODEL AND CLASS NAMES
# --------------------------------------------------

model.save(OUTPUT_DIR / "industrial_defect_cnn.keras")

with open(OUTPUT_DIR / "class_names.txt", "w") as file:
    file.write("\n".join(class_names))

print("\nModel saved in:", OUTPUT_DIR)
print("Training completed.")

