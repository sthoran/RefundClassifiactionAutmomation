import os
import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Enable MLflow autologging
mlflow.tensorflow.autolog()

# 📌 Define Paths (Relative)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get script directory
DATA_DIR = os.path.join(BASE_DIR, "apparel_images_dataset")  # Main dataset directory
CSV_PATH = os.path.join(BASE_DIR, "apparel_images_train.csv")  # CSV with relative paths
MODEL_DIR = os.path.join(BASE_DIR, "models")  # Where to save trained model

# 📌 Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

# 📌 Load dataset
df = pd.read_csv(CSV_PATH)

# ✅ No need to modify paths, keep them as relative
df["filepath"] = df["filepath"].astype(str)  # Ensure file paths are strings

# 📌 Split into train & test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# 📌 Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 📌 Define Image Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,  
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 📌 Create Data Generators using **relative paths**
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=BASE_DIR,  # 🔴 Base directory ensures images are loaded correctly
    x_col="filepath",  # Paths remain unchanged (relative to BASE_DIR)
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse"
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=BASE_DIR,  # 🔴 Ensures images load from correct subfolders
    x_col="filepath",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=False
)

# 📌 Load Pretrained ResNet50 model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model for transfer learning

# 📌 Build the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output_layer = Dense(len(train_generator.class_indices), activation="softmax")(x)

# 📌 Compile the model
model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 📌 Print model summary
model.summary()

# 📌 Start MLflow experiment
with mlflow.start_run():
    mlflow.log_param("model", "ResNet50")
    
    # Train the model
    history = model.fit(train_generator, validation_data=test_generator, epochs=10)

    # Log final accuracy
    mlflow.log_metric("final_train_accuracy", history.history["accuracy"][-1])
    mlflow.log_metric("final_train_loss", history.history["loss"][-1])

    # Save the trained model
    model_path = os.path.join(MODEL_DIR, "apparel_classifier_resnet50.h5")
    model.save(model_path)
    
    mlflow.tensorflow.log_model(model, artifact_path="models/ResNet50")

print("✅ Training complete! Model saved and logged in MLflow.")

# 📌 Plot Training Curves
plt.figure(figsize=(12, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")

plt.show()
