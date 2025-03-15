from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import os
from PIL import Image

# Define relative paths
DATA_DIR = os.path.relpath("test_data")
CSV_PATH = os.path.relpath("apparel_images_test.csv") 
MODEL_PATH = os.path.relpath("models/ResNet50_v2.h5")

# Load the CNN model
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Class labels
CLASS_NAMES = [
    'black_dress', 'black_pants', 'black_shirt', 'black_shoes', 'black_shorts',
    'blue_dress', 'blue_pants', 'blue_shirt', 'blue_shoes', 'blue_shorts',
    'brown_pants', 'brown_shoes', 'brown_shorts', 'green_pants', 'green_shirt',
    'green_shoes', 'green_shorts', 'red_dress', 'red_pants',
    'red_shoes', 'white_dress', 'white_pants', 'white_shoes', 'white_shorts'
]

# Set the start date
START_DATE = datetime.date(2025, 1, 1)

# Initialize FastAPI app
app = FastAPI()

# Global variables for tracking
CURRENTLY_PROCESSING = START_DATE  
IMAGES_PROCESSED_COUNT = 0  

# Image preprocessing function
def preprocess_image(image_path):
    """Preprocess an image before classification."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))  
        img = np.array(img) / 255.0  
        img = np.expand_dims(img, axis=0)  
        return img
    except:
        return None

# Image classification function
def classify_images(target_date: str = None):
    """Classifies images and ensures all dates are processed sequentially."""
    global CURRENTLY_PROCESSING, IMAGES_PROCESSED_COUNT

    try:
        df = pd.read_csv(CSV_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date

        # Determine the date to process
        if target_date:
            next_date = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
        else:
            next_date = CURRENTLY_PROCESSING

        today = datetime.datetime.now().date()

        # Ensure processing happens in sequence without skipping days
        if next_date <= today:
            CURRENTLY_PROCESSING = next_date  
            IMAGES_PROCESSED_COUNT = 0  

            day_images = df[df['timestamp'] == next_date]

            if not day_images.empty:
                predictions = []
                for _, row in day_images.iterrows():
                    image_path = row['filepath']
                    img = preprocess_image(image_path)
                    if img is None:
                        continue  
                    pred = model.predict(img)[0]
                    top_class_idx = np.argmax(pred)
                    top_class_name = CLASS_NAMES[top_class_idx]
                    top_confidence = round(float(pred[top_class_idx]) * 100, 2)

                    predictions.append(f"{top_class_name} ({top_confidence}%)")
                    IMAGES_PROCESSED_COUNT += 1  

                if len(predictions) == len(day_images):
                    df.loc[df['timestamp'] == next_date, 'predicted_label'] = predictions
                    df.to_csv(CSV_PATH, index=False)

            CURRENTLY_PROCESSING += datetime.timedelta(days=1)  
        
        return {
            "message": "Classification completed.",
            "images_processed_count": IMAGES_PROCESSED_COUNT  
        }

    except Exception as e:
        CURRENTLY_PROCESSING = None
        return {
            "message": f"Error: {str(e)}",
            "images_processed_count": IMAGES_PROCESSED_COUNT  
        }

# Initialize APScheduler (Midnight Trigger)
scheduler = BackgroundScheduler()
scheduler.add_job(classify_images, "cron", hour=0, minute=0, id="midnight_trigger")
scheduler.start()

# API Endpoints
@app.get("/")
def home():
    """Check if the API is running."""
    return {"message": "CNN Image Classification API is running!"}

@app.get("/next_date")
def get_next_date():
    """Check the next scheduled processing date."""
    return {"next_processing_date": (CURRENTLY_PROCESSING + datetime.timedelta(days=1)).strftime('%Y-%m-%d')}

@app.post("/trigger")
def manual_trigger(target_date):
    """Manually trigger classification for a specific image."""
    result = classify_images(target_date=CURRENTLY_PROCESSING.strftime("%Y-%m-%d"))
    return result

# 1-Minute Trigger (Only for Testing)
@app.post("/start_1min_trigger")
def start_test_trigger():
    """Starts a test trigger to classify one day's images every 1 minute."""
    global CURRENTLY_PROCESSING

    # Ensure CURRENTLY_PROCESSING is initialized
    if CURRENTLY_PROCESSING is None:
        CURRENTLY_PROCESSING = START_DATE  

    def process_next_day():
        global CURRENTLY_PROCESSING
        classify_images(target_date=CURRENTLY_PROCESSING.strftime("%Y-%m-%d"))
        CURRENTLY_PROCESSING += datetime.timedelta(days=1)  

    # Schedule the process every 1 minute
    scheduler.add_job(process_next_day, "interval", minutes=1, id="test_trigger", replace_existing=True)
    
    return {"message": "1-minute test trigger activated!"}

@app.get("/processing_status")
def get_processing_status():
    """Returns the currently processing date and number of images classified."""
    return {
        "currently_processing": CURRENTLY_PROCESSING if CURRENTLY_PROCESSING else "No batch processing running",
        "images_processed_last_run": IMAGES_PROCESSED_COUNT
    }

@app.get("/list_jobs")
def list_jobs():
    """Lists all scheduled jobs."""
    return {"jobs": [str(job) for job in scheduler.get_jobs()]}

@app.post("/stop_1min_trigger")
def stop_test_trigger():
    """Stops the 1-minute test trigger."""
    scheduler.remove_job("test_trigger")
    return {"message": "1-minute test trigger stopped!"}
