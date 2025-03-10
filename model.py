#%%
import pandas as pd
import numpy as np
import cv2
import h5py
import os
import tqdm
from collections import Counter

# %% convert data to HDF5 Format
image_folder = '/Users/user/IUBH/Semester3/frommodeltoproduction/1/images_compressed'
csv_path = '/Users/user/IUBH/Semester3/frommodeltoproduction/1/images.csv'
hdf5_path = 'Dataset.h5'

# %%
df = pd.read_csv(csv_path)
# %%
df.head()
# %% add datetime metadata

# Shuffle the dataset to ensure randomness
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
number_entries = len(df)
# Generate date range
start_date = pd.Timestamp("2025-01-01")  # Change this to your desired start date
num_days = (number_entries // 150) + 1  # Calculate the required number of days
dates = pd.date_range(start=start_date, periods=num_days, freq='D')

# Assign dates to chunks of 150
df['date'] = np.repeat(dates, 150)[:number_entries]

df.to_csv(csv_path)


#%% check for duplicates
# Count occurrences of image names in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.JPG', '.png'))]
duplicates = [item for item, count in Counter(image_files).items() if count > 1]

# %% convert data to HDF5 format
# Create HDF5 file
with h5py.File(hdf5_path, "w") as f:
    # Store labels
    labels = df["label"].values
    f.create_dataset("labels", data=labels)

    # First image to determine shape
    image_path = '/Users/user/IUBH/Semester3/frommodeltoproduction/1/images_compressed/ce8d7054-ee06-4412-b5df-a610943e3e50.jpg'
    first_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_shape = first_img.shape  # (Height, Width, Channels)

    #Create HDF5 dataset for images
    img_dtype = np.uint8  # Store images as 8-bit integers
    image_dataset = f.create_dataset("images", shape=(len(df), *img_shape), dtype=img_dtype, compression="gzip")

    # Loop through images and store in HDF5
    for i, img_id in tqdm.tqdm(enumerate(df["image"]), total=len(df)):
        img_path = os.path.join(image_folder, img_id)+".jpg"
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Read image as BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        # Ensure image shape is consistent (resize if necessary)
        if img.shape != img_shape:
            img = cv2.resize(img, (img_shape[1], img_shape[0]))

        # Store image in dataset
        image_dataset[i] = img

print("Dataset successfully stored in HDF5!")
# %%
