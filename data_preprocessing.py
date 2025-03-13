#%% Import Necessary Libraries
import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import random
from sklearn.model_selection import train_test_split

#%% **Step 1: Define Dataset Path Dynamically**
project_root = os.getcwd()  # Gets the current project directory

# Allow user to specify dataset path, otherwise use default
dataset_path = os.path.join(project_root, "apparel_images_dataset")  

# Check if dataset exists before proceeding
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f" Dataset path not found: {dataset_path}\nMake sure 'apparel_images_dataset' exists!")

print(f"Dataset path found: {dataset_path}")

#%% **Step 2: Store Image Paths and Labels in DataFrame (Using Relative Paths)**
data = []

# Loop through folders and store file paths, labels
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if os.path.isdir(label_path): 
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            relative_img_path = os.path.relpath(img_path, project_root)  # Convert to relative path
            data.append([relative_img_path, label])  # Store relative paths

# Convert to DataFrame
df = pd.DataFrame(data, columns=["filepath", "label"])

#%% **Step 3: Check Class Distribution**
class_counts = df['label'].value_counts()

# Plot class distribution
plt.figure(figsize=(12, 6))
class_counts.plot(kind="bar", color="skyblue")
plt.title("Class Distribution Before Augmentation")
plt.xlabel("Categories")
plt.ylabel("Number of Images")
plt.xticks(rotation=90)
plt.show()

#%% **Step 4: Perform Data Augmentation**
# Define augmentation parameters
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Track new augmented data
augmented_data = []
min_images = 500  # Minimum number of images per class

for label, count in class_counts.items():
    if count < min_images:
        needed = min_images - count  
        label_dir = os.path.join(dataset_path, label)
        images = os.listdir(label_dir)

        for i in range(needed):
            img_name = random.choice(images)  # Pick a random existing image
            img_path = os.path.join(label_dir, img_name)

            # Load and preprocess image
            img = Image.open(img_path)
            img = img.resize((224, 224))  # Resize for consistency
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array.reshape((1,) + img_array.shape)  # Reshape for augmentation

            # Extract original filename without extension
            original_name, ext = os.path.splitext(img_name)
            save_prefix = f"aug_{original_name}"
            save_format = ext[1:]

            for batch in datagen.flow(img_array, batch_size=1, save_to_dir=label_dir, save_prefix=save_prefix, save_format=save_format):
                break  # Generate only one image per iteration
            
            # Find the newly saved augmented image
            new_augmented_file = None
            for file in os.listdir(label_dir):
                if file.startswith(f"aug_{original_name}") and file.endswith(ext):
                    new_augmented_file = os.path.join(label_dir, file)
                    break  # Take the first match

            if new_augmented_file:
                relative_augmented_path = os.path.relpath(new_augmented_file, project_root)  # Convert to relative
                augmented_data.append([relative_augmented_path, label])

# Convert augmented data to DataFrame
df_augmented = pd.DataFrame(augmented_data, columns=["filepath", "label"])

#%% **Step 5: Append Augmented Data to Original DataFrame**
df_final = pd.concat([df, df_augmented], ignore_index=True)

# Shuffle the dataset to ensure randomness
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

#%% **Step 6: Assign Timestamps in Batches of 200 (Randomized)**
start_date = datetime(2025, 1, 1)  # Initialize start date

timestamps = []
for i in range(0, len(df_final), 200):
    batch_date = start_date + timedelta(days=i // 200)
    batch_timestamp = batch_date.strftime("%Y-%m-%d")
    timestamps.extend([batch_timestamp] * min(200, len(df_final) - i))

# Add timestamp column to DataFrame
df_final["timestamp"] = timestamps

#%% **Step 7: Save Final Dataset with Relative Paths**
csv_filename = "apparel_images_balanced_with_dates.csv"
df_final.to_csv(csv_filename, index=False)
print(f"Augmented data saved! Total dataset size: {len(df_final)}")

#%% **Step 8: Read CSV and Convert Back to Full Paths for Use**
df_final = pd.read_csv(csv_filename)

print(df_final.head())  # Check the loaded paths

# Split into train (80%) and test (20%) sets
train_df, test_df = train_test_split(df_final, test_size=0.2, random_state=42, stratify=df["label"])

# Save the subsets as CSV files
train_csv = "apparel_images_train.csv"
test_csv = "apparel_images_test.csv"

train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)
#%% **Step 9: Check Final Class Distribution**
class_counts_final = df_final['label'].value_counts()

# Plot final class distribution
plt.figure(figsize=(12, 6))
class_counts_final.plot(kind="bar", color="skyblue")
plt.title("Class Distribution After Augmentation")
plt.xlabel("Categories")
plt.ylabel("Number of Images")
plt.xticks(rotation=90)
plt.show()

# %%
