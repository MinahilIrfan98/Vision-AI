# Phone Usage Detector (YOLO26)

This project uses the YOLO26 model to detect if a person is using a cell phone in an image.

## How it works
The program detects both "persons" and "cell phones". If a cell phone's bounding box is located inside a person's bounding box, the person is marked as "Using phone" with a red box. Otherwise, they get a green box.

## Setup Instructions
1. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```
2. **Activate the Environment**:
   - Windows: `.\venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Option 1: Command Line (CLI)
- Place your images in the `images/` folder.
- Run: `python main.py`
- Check the `outputs/` folder for the results.

### Option 2: Web Interface (UI)
- Run the following command in your terminal:
  ```bash
  streamlit run app.py
  ```
- The app will open automatically in your browser.

## Features
- **Auto-Detection**: Detects all 80 COCO classes by default.
- **Intelligent Filtering**: Option to hide specific classes you don't need to see.
- **Smart UI**: Sleek, modern interface with a responsive "chip" layout for detected objects.
- **Image Analysis**: Upload photos for side-by-side comparison and object counting.
- **Video Analysis**: Upload a video to track unique individuals and analyze phone usage patterns over time.
- **Detailed Analytics**: Get a per-person breakdown of phone usage percentage and a temporal chart of phone users.
- **Customizable Detection**: Adjust confidence thresholds and visualize confidence scores on labels.

## Limitations
- **False Positives**: A phone resting on a table near a person's bounding box may be incorrectly marked as "Using phone".
- **Missed Detections**: Very small phones or phones held far from the camera may be missed by the AI.
- **Video Playback**: Depending on your browser, you may need `ffmpeg` installed on your system for full video compatibility.


# Vision-AI
