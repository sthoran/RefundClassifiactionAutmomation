#%%
import os
import pandas as pd
import numpy as np
#import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
#%%
# Enable MLflow autologging
mlflow.tensorflow.autolog()

# Define Paths (Relative)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
DATA_DIR = os.path.join(BASE_DIR, "apparel_images_dataset") 
df_train = pd.read_csv('apparel_images_train.csv')
#test_CSV_PATH = os.path.join(BASE_DIR, 'apparel_images_test.csv')
MODEL_DIR = os.path.join(BASE_DIR, "models")  
# Ensure file paths in CSV are relative to `DATA_DIR`
df_train["filepath"] = df_train["filepath"].str.replace("apparel_images_dataset/", "", regex=False)
#%%
df_train.head()
#%%
# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
#%%
# Define Image Data Generators
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
#%%
# Create Data Generators using **relative paths**
train_generator = train_datagen.flow_from_dataframe(
    dataframe= df_train,
    directory=DATA_DIR,  
    x_col="filepath",  
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse"
)
#%%
# Load Pretrained ResNet50 model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

# Build the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output_layer = Dense(len(train_generator.class_indices), activation="softmax")(x)
#%%
# Compile the model
model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Print model summary
model.summary()

# Start MLflow experiment
with mlflow.start_run():
    mlflow.log_param("model", "ResNet50")
    
    # Train the model
    history = model.fit(train_generator, epochs=10)
#%%
    # Log final accuracy
    mlflow.log_metric("final_train_accuracy", history.history["accuracy"][-1])
    mlflow.log_metric("final_train_loss", history.history["loss"][-1])

    # Save the trained model
    model_path = os.path.join(MODEL_DIR, "apparel_classifier_resnet50.h5")
    model.save(model_path)
    
    mlflow.tensorflow.log_model(model, artifact_path="models/ResNet50")

print("Training complete! Model saved and logged in MLflow.")

# Plot Training Curves
plt.figure(figsize=(12, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training Accuracy")

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss")

plt.show()
