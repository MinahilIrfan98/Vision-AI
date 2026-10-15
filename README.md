# VisionAI — Phone Usage Detection

VisionAI is an image-analysis application that uses a YOLO object-detection model to identify people, mobile phones, and other objects. Its Streamlit interface lets you upload images, review annotated detections, adjust model confidence, filter classes, and download the processed images.

## Features

- Upload one or multiple JPG, JPEG, or PNG images.
- Detect objects from the model's supported COCO classes.
- Highlight people and mobile phones, and flag a person when a detected phone falls inside their bounding box.
- Adjust the confidence threshold and hide selected classes.
- Optionally show confidence scores on detection labels.
- Review object counts, detected classes, and a visual summary.
- Download each annotated image as a JPEG.
- Switch between the original and annotated image in the results view.

## Project structure

```text
.
├── app.py          # Streamlit web application
├── main.py         # YOLO inference and image annotation logic
├── requirements.txt
├── logo.svg        # Application logo
├── yolo26n.pt      # YOLO model weights used by the app
└── yolov8n.pt      # Additional model weights included with the project
```

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer recommended
- Internet access the first time Python dependencies are installed

The application loads `yolo26n.pt` from the project directory. Keep this model file beside `app.py` when running the application.

## Installation

Open a terminal in the project directory. In VS Code, use **Terminal → New Terminal**.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it using the command for your terminal:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Windows Git Bash
source .venv/Scripts/activate
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the application

From the project directory, with the virtual environment active, run:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal (usually <http://localhost:8501>). Upload one or more images, adjust the detection settings if needed, and download annotated results beneath each image.

To stop the application, focus the terminal and press **Ctrl+C**.

## How phone usage is classified

The application uses the model's `person` and `cell phone` detections. A phone is associated with a person when the phone box's center is inside that person's bounding box. The person is then labeled **Using phone**. This is a simple spatial heuristic and does not establish intent or confirm that the phone is being held.

## Troubleshooting

- **`streamlit` is not found:** Activate the virtual environment and run `python -m streamlit run app.py`.
- **PowerShell blocks activation:** In Git Bash, use `source .venv/Scripts/activate`; alternatively, invoke the environment's Python directly: `.venv/Scripts/python.exe -m streamlit run app.py`.
- **Model file not found:** Confirm `yolo26n.pt` is in the project directory alongside `app.py`.
- **No phone detected:** Try a clearer, higher-resolution image and lower the confidence threshold. Small, occluded, or distant phones may be missed by the model.
- **A nearby phone is associated with a person:** The current rule checks whether the phone's center falls inside a person box, so overlapping boxes can sometimes produce a false association.

## Notes

- Images are processed by the local application while it is running; the app does not include user accounts or a database.
- Detection results depend on image quality and model performance.
- This project currently provides image analysis through the web interface; video analysis is not included in the app.

## License

Add the license that applies to this project and its model weights before redistributing it. Review the model and dependency licenses separately.
