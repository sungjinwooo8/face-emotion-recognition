# Face & Emotion Recognition System

A comprehensive face recognition and emotion detection system built with OpenCV, TensorFlow, and Streamlit.

## Features

- 📸 **Collect Faces**: Capture and save face images for training
- 🎓 **Train Model**: Train face recognition model on collected data
- 🔍 **Recognition**: Real-time face recognition and emotion detection
- 📊 **Statistics**: View dataset and model statistics

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the emotion model file:
   - `emotion_model_IIITM.h5` should be in the project root directory

## Running the Streamlit App

To run the Streamlit frontend:

**Windows:**
- Double-click `START_APP.bat`, or
- Run: `streamlit run app.py`

**Linux/Mac:**
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage

### 1. Collect Faces
1. Navigate to "📸 Collect Faces" page
2. Enter a person's name
3. Use the camera to capture face images
4. Click "Save Image" for each capture
5. Aim for 20-50 images per person for best results

### 2. Train Model
1. Navigate to "🎓 Train Model" page
2. Review the dataset statistics
3. Click "Train Model" to train the face recognizer
4. Wait for training to complete

### 3. Recognition
1. Navigate to "🔍 Recognition" page
2. Use the camera to capture an image
3. View face recognition and emotion detection results

### 4. Statistics
1. Navigate to "📊 Statistics" page
2. View dataset and model status

## File Structure

```
face_emotion_project/
├── app.py                      # Streamlit frontend application (main app)
├── collect_faces.py            # Original face collection script
├── train_face_recognizer.py    # Original training script
├── live_face_and_emotion.py    # Original recognition script
├── emotion_model_IIITM.h5      # Emotion detection model
├── face_recognizer.yml         # Trained face recognition model (generated)
├── labels.pickle               # Face label mappings (generated)
├── faces_dataset/              # Face images directory (generated)
├── requirements.txt            # Python dependencies
├── START_APP.bat               # Quick launcher for Windows
└── README.md                   # This file
```

## Dependencies

- streamlit>=1.28.0
- opencv-python>=4.8.0
- numpy>=1.24.0
- tensorflow>=2.13.0
- Pillow>=10.0.0

## Notes

- The `faces_dataset/` directory contains collected face images (excluded from git)
- Model files (`*.yml`, `*.pickle`, `*.h5`) may be large and are excluded from git
- For best results, collect diverse face images under different lighting conditions and angles

## License

This project is for educational purposes.

