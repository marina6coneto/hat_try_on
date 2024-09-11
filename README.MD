
# Hat Try-On App

This project is a real-time Hat Try-On application using OpenCV, Python, and Haar Cascade for face detection. The application allows users to overlay different hat images on their faces captured through a webcam.

## Features

- Real-time face detection.
- Select different hats from a sidebar by clicking.
- Navigate through hats using keyboard shortcuts (`n` for next hat, `p` for previous hat).
- Hats are resized and positioned according to the detected face dimensions.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Pillow

You can install the required libraries using the following command:

```bash
pip install opencv-python numpy pillow
```

## How to Run the Project

1. Clone this repository.
2. Ensure the hat images are in a folder named `hats` in the same directory as the script.
3. Run the Python script:

```bash
python main.py
```

## How to Use

- The app will automatically open your webcam.
- The sidebar on the right shows different hats available for overlay. 
- Click on a hat to select it or use the keyboard shortcuts (`n` for next, `p` for previous).
- Press `q` to quit the application.