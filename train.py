import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import cv2
import shutil


#UNZIP DATASET

ZIP_PATH = "dataset.zip"
DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    print("🔍 Extracting dataset.zip ...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall("data")

print("✅ Dataset extracted.")


#CLEAN AND PREPROCESS DATASET


IMG_SIZE = 96
OUTPUT_DIR = "processed_data"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

splits = ["train", "val", "test"]


for split in splits:
    input_path = os.path.join(DATA_DIR, split)
    output_path = os.path.join(OUTPUT_DIR, split)

    os.makedirs(output_path, exist_ok=True)

    class_folders = sorted(os.listdir(input_path))
    for cls in class_folders:
        in_folder = os.path.join(input_path, cls)
        out_folder = os.path.join(output_path, cls)
        os.makedirs(out_folder, exist_ok=True)

        images = os.listdir(in_folder)
        for img_name in tqdm(images, desc=f"{split}/{cls}"):
            img_path = os.path.join(in_folder, img_name)

            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # convert grayscale to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # resize to 96x96
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                # save processed image
                save_path = os.path.join(out_folder, img_name)
                cv2.imwrite(save_path, img)

            except:
                pass

print("✅ Preprocessing complete.\n")


#LOAD DATASET

BATCH_SIZE = 64

train_ds = keras.utils.image_dataset_from_directory(
    f"{OUTPUT_DIR}/train",
    labels="inferred",
    label_mode="int",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = keras.utils.image_dataset_from_directory(
    f"{OUTPUT_DIR}/val",
    labels="inferred",
    label_mode="int",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    f"{OUTPUT_DIR}/test",
    labels="inferred",
    label_mode="int",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)


# Improve performance
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


#BUILD CNN MODEL


def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = layers.Rescaling(1./255)(inputs)

    #augmentation
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomContrast(0.2)
    ])
    x = aug(x)

    # BLOCK 1
    x = layers.Conv2D(32, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(32, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPooling2D()(x)

    # BLOCK 2
    x = layers.Conv2D(64, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPooling2D()(x)

    # BLOCK 3
    x = layers.Conv2D(128, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPooling2D()(x)

    # FC layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(len(class_names), activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model


model = build_model()
model.summary()

#COMPILE MODEL


model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)



#ADD EARLY STOPPING / REDUCE LR

callbacks = [
    keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="models/emotion_best.keras",
        save_best_only=True,
        monitor="val_accuracy",
        mode="max"
    )
]

#TRAIN

EPOCHS = 25
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

#TEST ACCURACY

print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n🎉 Test Accuracy: {test_acc:.4f}")
