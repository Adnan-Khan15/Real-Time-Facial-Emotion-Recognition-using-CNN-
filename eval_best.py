import tensorflow as tf
from tensorflow import keras

best_model_path = "models/emotion_best.keras"   # or models/best_model.h5 depending on your code
model = keras.models.load_model(best_model_path)
print("Loaded model:", best_model_path)

# evaluate on test set (assumes test_ds exists or use image_dataset_from_directory)
test_ds = keras.utils.image_dataset_from_directory(
    "processed_data/test", image_size=(96,96), batch_size=64, label_mode="int", shuffle=False
)
loss, acc = model.evaluate(test_ds)
print("Test loss:", loss, "Test acc:", acc)
