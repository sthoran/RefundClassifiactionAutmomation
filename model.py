#%%
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
#%%
# Enable MLflow autologging
mlflow.tensorflow.autolog()

# Define Paths (Relative)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
DATA_DIR = os.path.join(BASE_DIR, "apparel_images_dataset") 
df_train = pd.read_csv('apparel_images_train.csv')
MODEL_DIR = os.path.join(BASE_DIR, "models")  
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
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    shear_range=0.2,
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
base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[100:]:  
    layer.trainable = False 

# Build the Sequential Model
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(24, activation="softmax") 
])
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0010000000474974513,  
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False
)

# Recompile the model
model.compile(
    optimizer=optimizer,
    loss=   "sparse_categorical_crossentropy", 
    metrics=["accuracy"])
#%%
# Print model summary
model.summary()

model_name = "ResNet50_v3" 
model_path = os.path.join(MODEL_DIR, f"{model_name}.h5")
#%%
# Start MLflow experiment
with mlflow.start_run():
    mlflow.log_param("model_name", model_name)
    
    # Train the model
    history = model.fit(train_generator, epochs=10)
#%%   
    # Log final accuracy
    mlflow.log_metric("final_train_accuracy", history.history["accuracy"][-1])
    mlflow.log_metric("final_train_loss", history.history["loss"][-1])

    model.save(model_path)
    input_example = np.random.rand(1, 224, 224, 3).astype(np.float32)  # Example of a normalized image

# Log model with input example
mlflow.tensorflow.log_model(
    model,
    artifact_path=f"models/{model_name}",
    input_example=input_example )
    
print("Training complete! Model saved and logged in MLflow.")
mlflow.end_run()
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

# %%
