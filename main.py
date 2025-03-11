#%%
from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import logging
from PIL import Image
import os
#%%
# Paths for mounted data
DATA_DIR = "/app/data"  
CSV_PATH = os.path.join(DATA_DIR, "test_data.csv") 

# Load the CNN model
MODEL_PATH = os.path.join(DATA_DIR, "apparel_classifier_resnet50.h5")  
model = tf.keras.models.load_model(MODEL_PATH)

#%%
# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
#%%
# Class labels
CLASS_NAMES = [
    'black_dress', 'black_pants', 'black_shirt', 'black_shoes', 'black_shorts',
    'blue_dress', 'blue_pants', 'blue_shirt', 'blue_shoes', 'blue_shorts',
    'brown_pants', 'brown_shoes', 'brown_shorts', 'green_pants', 'green_shirt',
    'green_shoes', 'green_shorts', 'red_dress', 'red_pants',
    'red_shoes', 'white_dress', 'white_pants', 'white_shoes', 'white_shorts'
]
#%%
# Start date for processing
START_DATE = datetime.datetime.strptime("20250101", "%Y%m%d").date()
#%%
# Initialize FastAPI app
app = FastAPI()
#%%
# Image Preprocessing Function (Uses Mounted Data)
def preprocess_image(image_name):
    """Preprocess an image before classification."""
    image_path = os.path.join(DATA_DIR, "images", image_name) 
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))  # Resize for the CNN model
        img = np.array(img) / 255.0   # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None
#%%
# Global variables for real-time tracking
CURRENTLY_PROCESSING = None  # Tracks the day being processed
IMAGES_PROCESSED_COUNT = 0  # Tracks number of images classified
#%%
# Image Classification Function: Ensuring No Skipped Days
def classify_images(target_date: str = None):
    """Classifies images from test_data.csv and ensures all dates are processed sequentially."""
    global CURRENTLY_PROCESSING, IMAGES_PROCESSED_COUNT

    try:
        df = pd.read_csv(CSV_PATH)  
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date

        # Determine the next date to process
        if target_date:
            next_date = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
        else:
            last_processed_date = df[df['predicted_label'].notna()]['timestamp'].max() if 'predicted_label' in df.columns else START_DATE
            next_date = last_processed_date + datetime.timedelta(days=1)

        today = datetime.datetime.now().date()

        # Ensure processing happens in sequence without skipping days
        while next_date < today:
            logging.info(f"Processing images for {next_date.strftime('%Y-%m-%d')}")
            CURRENTLY_PROCESSING = next_date.strftime('%Y-%m-%d')  # Track current processing date
            IMAGES_PROCESSED_COUNT = 0  # Reset count for this batch

            day_images = df[df['timestamp'] == next_date]

            if day_images.empty:
                logging.warning(f"No images found for {next_date.strftime('%Y-%m-%d')}. Moving to next day.")
            else:
                predictions = []
                for _, row in day_images.iterrows():
                    image_name = os.path.basename(row['filepath'])  
                    img = preprocess_image(image_name)
                    if img is None:
                        continue  
                    pred = model.predict(img)[0]
                    top_class_idx = np.argmax(pred)
                    top_class_name = CLASS_NAMES[top_class_idx]
                    top_confidence = round(float(pred[top_class_idx]) * 100, 2)

                    predictions.append(f"{top_class_name} ({top_confidence}%)")
                    IMAGES_PROCESSED_COUNT += 1  # Track images classified

                # Update CSV with predictions
                df.loc[df['timestamp'] == next_date, 'predicted_label'] = predictions
                df.to_csv(CSV_PATH, index=False)  # 
                logging.info(f"Processed {IMAGES_PROCESSED_COUNT} images for {next_date.strftime('%Y-%m-%d')}.")

            next_date += datetime.timedelta(days=1)  # Move to next date

        CURRENTLY_PROCESSING = None  # Reset after completion

        return {"message": "All unprocessed days have been classified successfully."}

    except Exception as e:
        logging.error(f"Error during batch processing: {str(e)}")
        CURRENTLY_PROCESSING = None
        return {"message": f"Error: {str(e)}"}
#%%
# APScheduler Initialization (Midnight Trigger for Production)
scheduler = BackgroundScheduler()
scheduler.add_job(classify_images, "cron", hour=0, minute=0, id="midnight_trigger")
scheduler.start()
logging.info("APScheduler started for daily midnight batch processing.")
#%%
# API Endpoints

@app.get("/")
def home():
    """Check if the API is running."""
    return {"message": "CNN Image Classification API is running!"}

@app.get("/next_date")
def get_next_date():
    """Check the next scheduled processing date."""
    df = pd.read_csv(CSV_PATH)  
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
    last_processed_date = df[df['predicted_label'].notna()]['timestamp'].max() if 'predicted_label' in df.columns else START_DATE
    return {"next_processing_date": (last_processed_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')}

@app.post("/trigger")
def manual_trigger(date: str = None):
    """Manually trigger classification for a specific date."""
    result = classify_images(target_date=date)
    return {"message": result}

# 5-Minute Trigger (Only for Testing)
@app.post("/start_5min_trigger")
def start_test_trigger():
    """Manually starts the 5-minute batch processing test."""
    logging.info("5-minute batch processing trigger activated.")

    def process_next_day():
        global CURRENTLY_PROCESSING
        CURRENTLY_PROCESSING = datetime.datetime.now().strftime('%Y-%m-%d')  
        classify_images()
        CURRENTLY_PROCESSING = None  

    scheduler.add_job(process_next_day, "interval", minutes=5, id="test_trigger", replace_existing=True)
    return {"message": "5-minute test trigger activated!"}

@app.get("/processing_status")
def get_processing_status():
    """Returns the currently processing date and number of images classified."""
    return {
        "currently_processing": CURRENTLY_PROCESSING if CURRENTLY_PROCESSING else "No batch processing running",
        "images_processed_last_run": IMAGES_PROCESSED_COUNT
    }

@app.post("/stop_5min_trigger")
def stop_test_trigger():
    """Stops the manually activated 5-minute batch processing."""
    scheduler.remove_job("test_trigger")
    return {"message": "5-minute test trigger stopped!"}

