# CNN Image Classification API with Scheduler

This project is a FastAPI-based REST API that performs image classification using a pre-trained TensorFlow model.
It is designed to automatically classify daily image batches stored in an AWS S3 bucket and save the results locally to a CSV file. A test scheduler runs every minute for development/testing purposes.

---

## Features

- Automatically fetches and classifies daily images from S3
- Uses a ResNet50_v2 TensorFlow model loaded from S3
- BackgroundScheduler (APScheduler) runs classification jobs
- Saves predictions to a local `modified_predictions.csv` file
- Includes a 1-minute trigger for rapid testing
- API endpoints for triggering, checking status, and downloading results

---

##  Project Structure
. ├── main.py # FastAPI app and scheduler
├── modified_predictions.csv # Output file with predictions (generated) 
├── requirements.txt # Python dependencies
└── README.md # This file

---

## Getting Started

### 1. clone repository
```bash
git clone https://github.com/sthoran/RefundClassifiactionAutmomation.git ```

cd RefundClassifiactionAutmomation.git

### 2. install dependencies

```bash
pip install -r requirements.txt

### 3. Run API
```bash
uvicorn main:app --reload

### 4. open Swagger ui 

output:

Example INFO:     Uvicorn running on http://127.0.0.1:8000

type the following to access swagger ui in browser

``` browser
http://127.0.0.1:8000/docs

## API Endpoints

Method	Endpoint	Description
GET	/	Health check
GET	/next_date	Shows the next date to be processed
GET	/download_csv	Downloads the predictions CSV
GET	/list_jobs	Lists all running scheduled jobs
POST	/trigger?target_date=YYYY-MM-DD	Manually trigger classification for a given date
POST	/start_1min_trigger	Starts test trigger to classify daily images every 1 min
POST	/stop_1min_trigger	Stops the test trigger


## How scheduler works
- Starts processing from START_DATE (2025-01-01)
- Runs classify_images() once per minute (in test mode)
For each date:

    - Downloads the image metadata CSV from S3
    - Filters image paths for the current date
    - Downloads and preprocesses each image
    - Runs prediction using the loaded model
    - Saves results to modified_predictions.csv
    - Increments to the next date

## Notes

- Images are downloaded temporarily to /tmp and deleted after classification.
- All paths are built relative to the S3 structure, and row['filepath'] should already include the correct folder structure.
- If no images are found for a date, the scheduler will skip to the next day.




