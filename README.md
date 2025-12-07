<div align="center">

# 😊 Face & Emotion Recognition System

**A powerful, web-based face recognition and emotion detection system built with OpenCV, TensorFlow, and Streamlit**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52+-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Recognize faces in real-time and detect emotions with state-of-the-art AI technology**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Demo](#-demo) • [Tech Stack](#-tech-stack) • [Contributing](#-contributing)

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Screenshots](#-screenshots)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [How It Works](#-how-it-works)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project is a **comprehensive face recognition and emotion detection system** that combines the power of OpenCV for face detection, TensorFlow for emotion classification, and Streamlit for a beautiful web interface. It allows you to:

- 🎭 **Collect face data** through an intuitive web interface
- 🧠 **Train custom face recognition models** using LBPH (Local Binary Patterns Histograms)
- 👁️ **Recognize faces in real-time** from webcam input
- 😊 **Detect emotions** with high accuracy (neutral, sad, smile, surprise, yawning)
- 📊 **Monitor system statistics** and model performance

Perfect for security systems, attendance tracking, emotion analysis, and educational purposes!

---

## ✨ Features

### 🎨 User Interface
- **Beautiful Streamlit Web Interface** - Modern, responsive design
- **Multi-Page Navigation** - Easy-to-use sidebar navigation
- **Real-time Camera Integration** - Direct webcam access from browser
- **Live Preview** - See your face detection in real-time

### 🤖 Face Recognition
- **Custom Training** - Train models on your own face dataset
- **LBPH Algorithm** - Robust Local Binary Patterns Histograms recognition
- **Multiple Person Support** - Recognize multiple individuals
- **Confidence Scoring** - See recognition confidence levels

### 😊 Emotion Detection
- **6 Emotion Categories** - Neutral, Sad, Smile, Surprise, Surprise Open, Yawning
- **Deep Learning Model** - Pre-trained TensorFlow/Keras model
- **High Accuracy** - State-of-the-art emotion classification
- **Real-time Prediction** - Instant emotion detection

### 📸 Data Collection
- **Easy Capture** - One-click face image collection
- **Automatic Face Detection** - Smart face cropping and alignment
- **Bulk Collection** - Collect 20-50 images per person quickly
- **Dataset Management** - View and manage collected faces

### 📊 Analytics
- **Statistics Dashboard** - View dataset and model information
- **Training Progress** - Real-time training status updates
- **Model Status** - Check which models are loaded and ready

---

## 📸 Screenshots

> **Note:** Add screenshots of your app here to showcase the interface!

```
[📸 Collect Faces Page]     [🎓 Train Model Page]
[🔍 Recognition Page]       [📊 Statistics Page]
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- Webcam (for face collection and recognition)
- Windows/Linux/Mac OS

### Step 1: Clone the Repository

```bash
git clone https://github.com/sungjinwooo8/face-emotion-recognition.git
cd face-emotion-recognition
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `streamlit>=1.28.0` - Web framework
- `opencv-python>=4.8.0` - Computer vision
- `numpy>=1.24.0` - Numerical computing
- `tensorflow>=2.13.0` - Deep learning
- `Pillow>=10.0.0` - Image processing

### Step 4: Verify Emotion Model

Ensure you have the emotion detection model file:
- `emotion_model_IIITM.h5` should be in the project root directory

> **Note:** If you don't have the model file, emotion detection won't work, but face recognition will still function after training.

---

## ⚡ Quick Start

### Windows Users

**Option 1: Double-click launcher**
```
Double-click START_APP.bat
```

**Option 2: Command line**
```powershell
streamlit run app.py
```

### Linux/Mac Users

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### 1️⃣ Collect Faces 📸

**Purpose:** Build your face recognition dataset

1. Navigate to the **"📸 Collect Faces"** page in the sidebar
2. Enter the person's name (no spaces recommended)
3. Allow camera access when prompted
4. Position yourself in front of the camera
5. Click **"📥 Save Image"** to capture each face
6. Repeat for 20-50 images per person for best accuracy
7. Try different angles, expressions, and lighting conditions

**Tips:**
- ✅ Good lighting helps with accuracy
- ✅ Capture faces from different angles
- ✅ Include various expressions
- ❌ Avoid extreme lighting (too dark/bright)
- ❌ Don't move too quickly during capture

### 2️⃣ Train Model 🎓

**Purpose:** Train the face recognition model on your collected data

1. Navigate to the **"🎓 Train Model"** page
2. Review the dataset statistics:
   - Number of persons registered
   - Total images collected
3. Click **"🚀 Train Model"** button
4. Wait for training to complete (usually takes a few seconds to minutes)
5. You'll see a success message with label mappings

**What happens:**
- Model loads all collected face images
- LBPH algorithm trains on your dataset
- Model files are saved:
  - `face_recognizer.yml` - Trained model
  - `labels.pickle` - Person name mappings

### 3️⃣ Recognition 🔍

**Purpose:** Recognize faces and detect emotions in real-time

1. Navigate to the **"🔍 Recognition"** page
2. Allow camera access
3. Point the camera at a face
4. View the results:
   - **Face Recognition:** Person's name (if trained)
   - **Emotion:** Detected emotion with confidence
   - **Confidence Scores:** Recognition and emotion probabilities

**Features:**
- Works with multiple faces in frame
- Real-time processing
- Confidence scores for both face and emotion
- Visual bounding boxes and labels

### 4️⃣ Statistics 📊

**Purpose:** Monitor your dataset and model status

1. Navigate to the **"📊 Statistics"** page
2. View:
   - **Registered Persons:** List of all trained individuals
   - **Image Count:** Number of images per person
   - **Model Status:** Which models are loaded and ready
   - **System Information:** Dataset statistics

---

## 📁 Project Structure

```
face-emotion-recognition/
│
├── 📱 Frontend Application
│   ├── app.py                      # Main Streamlit application
│   └── START_APP.bat               # Quick launcher (Windows)
│
├── 🐍 Python Scripts
│   ├── collect_faces.py            # Original face collection script
│   ├── train_face_recognizer.py    # Original training script
│   └── live_face_and_emotion.py    # Original recognition script
│
├── 🤖 Models & Data
│   ├── emotion_model_IIITM.h5      # Emotion detection model (pre-trained)
│   ├── face_recognizer.yml         # Face recognition model (generated)
│   ├── labels.pickle               # Person label mappings (generated)
│   └── faces_dataset/              # Collected face images (generated)
│       └── [person_name]/
│           └── [images].jpg
│
├── 📦 Configuration
│   ├── requirements.txt            # Python dependencies
│   ├── .gitignore                  # Git ignore rules
│   └── README.md                   # This file
│
└── 📚 Documentation
    └── README.md                   # Project documentation
```

---

## 🛠️ Tech Stack

### Frontend
- **Streamlit** - Modern web framework for Python
- **HTML/CSS** - Custom styling and layouts

### Backend & AI
- **OpenCV** - Computer vision and face detection
  - Haar Cascade Classifiers for face detection
  - LBPH (Local Binary Patterns Histograms) for face recognition
- **TensorFlow/Keras** - Deep learning framework
  - Pre-trained CNN model for emotion classification
- **NumPy** - Numerical computing and array operations

### Languages & Tools
- **Python 3.11+** - Primary programming language
- **Pickle** - Model serialization
- **PIL/Pillow** - Image processing

---

## 🧠 How It Works

### Face Recognition Pipeline

```
1. Face Collection → 2. Preprocessing → 3. Training → 4. Recognition
```

1. **Collection:** Capture face images using webcam
2. **Preprocessing:** Convert to grayscale, resize to 200x200 pixels
3. **Training:** LBPH algorithm learns facial patterns
4. **Recognition:** Compare live faces against trained model

### Emotion Detection Pipeline

```
1. Face Detection → 2. Preprocessing → 3. Emotion Classification → 4. Result
```

1. **Detection:** Locate face in frame using Haar Cascades
2. **Preprocessing:** Convert to grayscale, resize to 48x48 pixels, normalize
3. **Classification:** Deep learning model predicts emotion probabilities
4. **Result:** Display emotion with confidence score

### Emotion Categories

| Emotion | Description | Use Case |
|---------|-------------|----------|
| 😐 Neutral | Normal, expressionless face | Default state |
| 😢 Sad | Downward mouth, drooping features | Mood detection |
| 😊 Smile | Happy, upward mouth curve | Positive emotion |
| 😮 Surprise | Wide eyes, raised eyebrows | Reaction detection |
| 😱 Surprise Open | Extreme surprise with open mouth | Strong reactions |
| 🥱 Yawning | Open mouth, tired expression | Fatigue detection |

---

## 🔧 Troubleshooting

### Common Issues

**❌ Camera not working**
- Check camera permissions in browser/system settings
- Ensure no other application is using the camera
- Try refreshing the Streamlit page

**❌ Model not loading**
- Verify `emotion_model_IIITM.h5` exists in project root
- Check file permissions
- Re-download the model if corrupted

**❌ Face not detected**
- Improve lighting conditions
- Move closer to camera
- Ensure face is clearly visible
- Try different camera angles

**❌ Recognition accuracy is low**
- Collect more training images (20-50 per person)
- Use diverse lighting conditions
- Capture faces from different angles
- Retrain the model

**❌ Installation errors**
- Ensure Python 3.11+ is installed
- Use virtual environment
- Update pip: `python -m pip install --upgrade pip`
- Install packages one by one if batch install fails

### Windows Long Path Error

If you encounter path length errors during installation:

```powershell
# Enable Windows Long Path Support (requires admin)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Then restart your computer.

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute

- 🐛 **Report Bugs** - Open an issue describing the problem
- 💡 **Suggest Features** - Share your ideas for improvements
- 📝 **Improve Documentation** - Fix typos, add examples, clarify instructions
- 💻 **Submit Code** - Fix bugs or add new features

### Contribution Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Style

- Follow PEP 8 Python style guide
- Add comments for complex logic
- Update documentation as needed
- Test your changes before submitting

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Educational Purpose:** This project is designed for educational and research purposes. Please respect privacy and obtain consent when using face recognition technology.

---

## 🙏 Acknowledgments

- **OpenCV** - For powerful computer vision tools
- **TensorFlow** - For deep learning capabilities
- **Streamlit** - For the amazing web framework
- **Haar Cascade Classifiers** - For face detection
- **LBPH Algorithm** - For face recognition

---

## 📞 Support & Contact

- **GitHub Issues** - [Report bugs or request features](https://github.com/sungjinwooo8/face-emotion-recognition/issues)
- **Repository** - [View source code](https://github.com/sungjinwooo8/face-emotion-recognition)

---

## 🎯 Future Enhancements

- [ ] Real-time video streaming support
- [ ] Multiple camera support
- [ ] Cloud deployment options
- [ ] Mobile app integration
- [ ] Advanced analytics dashboard
- [ ] Export recognition logs
- [ ] Face mask detection
- [ ] Age and gender estimation
- [ ] Batch processing capabilities
- [ ] API endpoint for integration

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Made with ❤️ using Python, OpenCV, TensorFlow, and Streamlit**

[⬆ Back to Top](#-face--emotion-recognition-system)

</div>
