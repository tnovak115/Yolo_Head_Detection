
# YOLO Head Detection and CVAT Export Script

## Overview
This project performs head detection in images using a two-stage approach. It first detects persons using the YOLO model and then detects head positions within the detected persons using another object detection model. The results are drawn on the images and exported in CVAT format. The script is designed to be run in a batch mode, processing multiple images from a directory structure.

## Key Features
1. **YOLO Person Detection**: Uses the YOLO model to detect persons in the image.
2. **Head Detection in Cropped Images**: Applies a head detection model to cropped person images.
3. **Bounding Box Filtering**: Filters out false positives based on confidence scores.
4. **Drawing and Saving Images**: Draws detected heads on the original image and saves the result.
5. **CVAT XML Export**: Generates annotations in CVAT format for use in annotation tools.

## Dependencies
This script requires the following Python packages:
- `cv2` (OpenCV)
- `numpy`
- `PIL` (Python Imaging Library)
- `tqdm` (for progress bars)
- `ultralytics` (for YOLO)
- `torchvision`
- `torch`
- `xml.etree.ElementTree` (for XML manipulation)

## Usage
1. **Setup the Directory Structure**:
    Organize the image you want to detect in this strcture:
    ```
    ├── test_data
    │   ├── folder1
    │   │   ├── image1.jpg
    │   │   ├── image2.png
    │   └── folder2
    │       ├── image3.jpeg
    ```

2. **Download Pre-trained Models**:
   - Download your pre-trained YOLO model (`.pt` file) and head detection model from Google Drive(https://drive.google.com/drive/folders/1GjiDxO9aWXTvQbKw-52DEWO8bFQH8Uzi?usp=drive_link).
   
3. **Modify the Script**:
   Update the following paths in the script to point to the downloaded model paths:
   - `yolo_model` path: Path to your downloaded YOLO model weights (e.g., `./yolo_weights/yolov8l.pt`).
   - `head_model` path: Path to your pre-trained head detection model (e.g., `./models/head_detection.model`).

4. **Set Up the Python Environment**:
   Create a new virtual environment using Conda and install dependencies:
   ```
   conda create -n [your_environment_name] python=3.10.14
   conda activate [your_environment_name]
   pip install -r requirements.txt
   ```

5. **Run the Script**:
   After setting up the environment and installing dependencies, run the script:
   ```
   python yolo_head_detection.py
   ```


## Output
The script will generate:
1. Cropped images containing only the detected persons.
2. Images with bounding boxes drawn around the detected heads.
3. CVAT format XML files containing annotations for detected heads.

## Troubleshooting
- **Image Not Found Error**: Ensure that the input image path is correct.
- **YOLO Detection Failure**: Make sure the YOLO model weights are correctly loaded.
