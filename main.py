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
#from fastapi.responses import FileResponse
#%%
# Get the current working directory where the script is located
PROJECT_DIR = '/Users/user/IUBH/Semester3/frommodeltoproduction'
# Define the correct paths relative to the project directory
DATA_DIR = '/Users/user/IUBH/Semester3/frommodeltoproduction/apparel_data'
CSV_PATH = "/Users/user/IUBH/Semester3/frommodeltoproduction/test_data.csv"
MODEL_PATH = "/Users/user/IUBH/Semester3/frommodeltoproduction/apparel_classifier_resnet50.h5"
#%%
# Check if the model file exists (optional, for debugging purposes)
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
#%%
# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

# Class labels
CLASS_NAMES = [
    'black_dress', 'black_pants', 'black_shirt', 'black_shoes', 'black_shorts',
    'blue_dress', 'blue_pants', 'blue_shirt', 'blue_shoes', 'blue_shorts',
    'brown_pants', 'brown_shoes', 'brown_shorts', 'green_pants', 'green_shirt',
    'green_shoes', 'green_shorts', 'red_dress', 'red_pants',
    'red_shoes', 'white_dress', 'white_pants', 'white_shoes', 'white_shorts'
]

# Set the start date for classification
START_DATE = datetime.datetime.strptime("2025-01-01", "%Y-%m-%d").date()
#%%
# Initialize FastAPI app
app = FastAPI()
#%%
# Global variables for real-time tracking
CURRENTLY_PROCESSING = START_DATE  # Start from 2025-01-01
IMAGES_PROCESSED_COUNT = 0  # Tracks number of images classified
#%%
# Image Preprocessing Function
def preprocess_image(image_path):
    """Preprocess an image before classification."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))  # Resize for the CNN model
        img = np.array(img) / 255.0   # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None

# Image Classification Function
def classify_images(target_date: str = None):
    """Classifies images from test_data.csv and ensures all dates are processed sequentially."""
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
            logging.info(f"Processing images for {next_date.strftime('%Y-%m-%d')}")
            CURRENTLY_PROCESSING = next_date  # Track current processing date
            IMAGES_PROCESSED_COUNT = 0  # Reset count for this batch

            day_images = df[df['timestamp'] == next_date]

            if day_images.empty:
                logging.warning(f"No images found for {next_date.strftime('%Y-%m-%d')}. Moving to next day.")
            else:
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
                    IMAGES_PROCESSED_COUNT += 1  # Increment image count for each processed image

                # Update CSV with predictions
                # **Fix**: Ensure predictions list matches the rows in DataFrame
                if len(predictions) == len(day_images):
                    df.loc[df['timestamp'] == next_date, 'predicted_label'] = predictions
                else:
                    logging.error("The number of predictions does not match the number of images for the date.")
                    return {
                        "message": "Error: Predictions count does not match image count.",
                        "images_processed_count": IMAGES_PROCESSED_COUNT
                    }
                df.to_csv(CSV_PATH, index=False)
                logging.info(f"Processed {IMAGES_PROCESSED_COUNT} images for {next_date.strftime('%Y-%m-%d')}.")

            # Increment date for the next round of processing 
            CURRENTLY_PROCESSING += datetime.timedelta(days=1)
            logging.info(f"Next day to process: {CURRENTLY_PROCESSING}")
        else:
            logging.warning(f"Processing for {next_date.strftime('%Y-%m-%d')} is skipped as it is in the future.")
        
        return {
            "message": "Classification completed.",
            "images_processed_count": IMAGES_PROCESSED_COUNT  # Return images processed count
        }

    except Exception as e:
        logging.error(f"Error during batch processing: {str(e)}")
        CURRENTLY_PROCESSING = None
        return {
            "message": f"Error: {str(e)}",
            "images_processed_count": IMAGES_PROCESSED_COUNT  # Return processed count in case of error
        }
#%%
# APScheduler Initialization (Midnight Trigger for Production)
scheduler = BackgroundScheduler()
scheduler.add_job(classify_images, "cron", hour=0, minute=0, id="midnight_trigger")
scheduler.start()
logging.info("Midnight trigger scheduled for daily batch processing.")
#%%
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
    """Manually trigger classification for a specific date."""
    result = classify_images(target_date=CURRENTLY_PROCESSING.strftime("%Y-%m-%d"))
    return result

# 1-Minute Trigger (Only for Testing)
@app.post("/start_1min_trigger")
def start_test_trigger():
    """Starts a test trigger to classify one day's images every 5 minutes."""
    global CURRENTLY_PROCESSING

    # Ensure CURRENTLY_PROCESSING is initialized
    if CURRENTLY_PROCESSING is None:
        CURRENTLY_PROCESSING = START_DATE

    logging.info(f" 1-minute test trigger started from {CURRENTLY_PROCESSING}")  

    def process_next_day():
        global CURRENTLY_PROCESSING

        logging.info(f"Classifying images for {CURRENTLY_PROCESSING}")  

        # Call classify_images() normally
        result = classify_images(target_date=CURRENTLY_PROCESSING.strftime("%Y-%m-%d"))
        logging.info(f"Classification result: {result}")

        # Move to the next day
        CURRENTLY_PROCESSING += datetime.timedelta(days=1)
        logging.info(f"Next day set to {CURRENTLY_PROCESSING}")

    # Schedule the process every 1 minutes
    scheduler.add_job(process_next_day, "interval", minutes=1, id="test_trigger", replace_existing=True)
    logging.info("1-minute test trigger scheduled successfully.")
    
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

@app.post("/stop_1-min_trigger")
def stop_test_trigger():
    """Stops the 1-minute test trigger."""
    scheduler.remove_job("test_trigger")
    return {"message": "1-minute test trigger stopped!"}



