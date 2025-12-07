import streamlit as st
import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
import time

# Page configuration
st.set_page_config(
    page_title="Face & Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3rem;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'training_status' not in st.session_state:
    st.session_state.training_status = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Title
st.markdown('<h1 class="main-header">😊 Face & Emotion Recognition System</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["📸 Collect Faces", "🎓 Train Model", "🔍 Recognition", "📊 Statistics"]
)

# Constants
DATASET_DIR = "faces_dataset"
FACE_MODEL_PATH = "face_recognizer.yml"
LABELS_PATH = "labels.pickle"
EMOTION_MODEL_PATH = "emotion_model_IIITM.h5"
EMOTIONS = ['neutral', 'sad', 'smile', 'surprise', 'surprise_open', 'yawning']
IMG_SIZE = 48

# ========== PAGE 1: COLLECT FACES ==========
if page == "📸 Collect Faces":
    st.header("📸 Collect Face Data")
    st.write("Capture face images for training the face recognition model.")
    
    person_name = st.text_input("Enter person's name (no spaces):", placeholder="e.g., John")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        img_file_buffer = st.camera_input("Take pictures of the person's face")
    
    with col2:
        st.subheader("Instructions")
        st.write("1. Enter the person's name")
        st.write("2. Click the camera to capture")
        st.write("3. Take multiple photos from different angles")
        st.write("4. Aim for 20-50 images per person")
        
        if st.button("📥 Save Image"):
            if not person_name:
                st.error("Please enter a person's name first!")
            elif img_file_buffer is not None:
                # Create directory for person
                person_dir = os.path.join(DATASET_DIR, person_name)
                os.makedirs(person_dir, exist_ok=True)
                
                # Convert image to OpenCV format
                bytes_data = img_file_buffer.getvalue()
                img_array = np.frombuffer(bytes_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # Detect face
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
                
                if len(faces) > 0:
                    # Get the largest face
                    face = max(faces, key=lambda rect: rect[2] * rect[3])
                    x, y, w, h = face
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (200, 200))
                    
                    # Save face image
                    count = len([f for f in os.listdir(person_dir) if f.endswith('.jpg')])
                    img_path = os.path.join(person_dir, f"{person_name}_{count}.jpg")
                    cv2.imwrite(img_path, face_roi)
                    
                    st.success(f"✅ Saved face image #{count + 1} for {person_name}!")
                    
                    # Display saved image
                    st.image(face_roi, caption=f"Saved face #{count + 1}", use_container_width=True)
                else:
                    st.warning("⚠️ No face detected in the image. Please try again.")
    
    # Show existing dataset
    if os.path.exists(DATASET_DIR):
        st.subheader("📁 Current Dataset")
        persons = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
        if persons:
            for person in persons:
                person_path = os.path.join(DATASET_DIR, person)
                image_count = len([f for f in os.listdir(person_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                st.write(f"👤 **{person}**: {image_count} images")

# ========== PAGE 2: TRAIN MODEL ==========
elif page == "🎓 Train Model":
    st.header("🎓 Train Face Recognition Model")
    st.write("Train the face recognizer using collected face images.")
    
    if not os.path.exists(DATASET_DIR):
        st.error("❌ No dataset directory found. Please collect some faces first!")
    else:
        persons = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
        if not persons:
            st.error("❌ No persons found in dataset. Please collect some faces first!")
        else:
            st.success(f"✅ Found {len(persons)} person(s) in dataset: {', '.join(persons)}")
            
            # Show dataset statistics
            total_images = 0
            for person in persons:
                person_path = os.path.join(DATASET_DIR, person)
                image_count = len([f for f in os.listdir(person_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                total_images += image_count
                st.write(f"👤 **{person}**: {image_count} images")
            
            st.info(f"📊 Total images: {total_images}")
            
            if st.button("🚀 Train Model", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔄 Loading images...")
                    progress_bar.progress(10)
                    
                    faces = []
                    labels = []
                    label_ids = {}
                    current_id = 0
                    
                    for person_name in persons:
                        person_path = os.path.join(DATASET_DIR, person_name)
                        if person_name not in label_ids:
                            label_ids[person_name] = current_id
                            current_id += 1
                        
                        person_id = label_ids[person_name]
                        
                        for fname in os.listdir(person_path):
                            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                                continue
                            
                            img_path = os.path.join(person_path, fname)
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is None:
                                continue
                            
                            img_resized = cv2.resize(img, (200, 200))
                            faces.append(img_resized)
                            labels.append(person_id)
                    
                    status_text.text("🔄 Preparing data...")
                    progress_bar.progress(40)
                    
                    faces = np.array(faces)
                    labels = np.array(labels)
                    
                    status_text.text(f"🔄 Training on {len(faces)} faces...")
                    progress_bar.progress(60)
                    
                    recognizer = cv2.face.LBPHFaceRecognizer_create()
                    recognizer.train(faces, labels)
                    
                    status_text.text("💾 Saving model...")
                    progress_bar.progress(90)
                    
                    recognizer.write(FACE_MODEL_PATH)
                    with open(LABELS_PATH, "wb") as f:
                        pickle.dump(label_ids, f)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Training complete!")
                    
                    st.success(f"✅ Model trained successfully!")
                    st.json(label_ids)
                    st.session_state.training_status = True
                    
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
                    status_text.empty()
                    progress_bar.empty()

# ========== PAGE 3: RECOGNITION ==========
elif page == "🔍 Recognition":
    st.header("🔍 Face & Emotion Recognition")
    st.write("Real-time face recognition and emotion detection.")
    
    # Load models
    @st.cache_resource
    def load_models():
        try:
            # Load emotion model
            if os.path.exists(EMOTION_MODEL_PATH):
                emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)
            else:
                return None, None, None, None
            
            # Load face recognizer
            if os.path.exists(FACE_MODEL_PATH) and os.path.exists(LABELS_PATH):
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read(FACE_MODEL_PATH)
                
                with open(LABELS_PATH, "rb") as f:
                    label_ids = pickle.load(f)
                
                id_to_name = {v: k for k, v in label_ids.items()}
            else:
                recognizer = None
                id_to_name = None
            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            
            return emotion_model, recognizer, id_to_name, face_cascade
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return None, None, None, None
    
    emotion_model, recognizer, id_to_name, face_cascade = load_models()
    
    if emotion_model is None:
        st.error("❌ Emotion model not found! Please ensure emotion_model_IIITM.h5 exists.")
    else:
        if recognizer is None:
            st.warning("⚠️ Face recognizer not found. Only emotion detection will work.")
        
        def preprocess_for_emotion(face_bgr):
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            x = gray.astype("float32") / 255.0
            x = np.expand_dims(x, axis=-1)
            x = np.expand_dims(x, axis=0)
            return x
        
        # Camera input
        img_file_buffer = st.camera_input("Capture image for recognition")
        
        if img_file_buffer is not None:
            # Convert to OpenCV format
            bytes_data = img_file_buffer.getvalue()
            img_array = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_rects = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
            )
            
            if len(faces_rects) > 0:
                results = []
                
                for (x, y, w, h) in faces_rects:
                    face_roi_color = img_rgb[y:y+h, x:x+w]
                    face_roi_gray = gray[y:y+h, x:x+w]
                    
                    # Face recognition
                    person_name = "Unknown"
                    face_conf = 0
                    if recognizer is not None:
                        face_gray_resized = cv2.resize(face_roi_gray, (200, 200))
                        label_id, conf = recognizer.predict(face_gray_resized)
                        if conf < 90:
                            person_name = id_to_name.get(label_id, "Unknown")
                            face_conf = 100 - conf  # Convert to percentage
                    
                    # Emotion prediction
                    face_bgr = img[y:y+h, x:x+w]
                    emo_input = preprocess_for_emotion(face_bgr)
                    emo_probs = emotion_model.predict(emo_input, verbose=0)[0]
                    emo_idx = int(np.argmax(emo_probs))
                    emo_label = EMOTIONS[emo_idx]
                    emo_conf = float(emo_probs[emo_idx]) * 100
                    
                    # Draw rectangle and text
                    cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = f"{person_name} | {emo_label}"
                    cv2.putText(img_rgb, text, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    results.append({
                        'name': person_name,
                        'emotion': emo_label,
                        'emotion_confidence': emo_conf,
                        'face_confidence': face_conf,
                        'face_roi': face_roi_color
                    })
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(img_rgb, caption="Recognition Result", use_container_width=True)
                
                with col2:
                    st.subheader("📊 Detection Results")
                    for i, result in enumerate(results, 1):
                        with st.container():
                            st.markdown(f"### Face {i}")
                            st.write(f"**Name:** {result['name']}")
                            if result['face_confidence'] > 0:
                                st.write(f"**Face Confidence:** {result['face_confidence']:.1f}%")
                            st.write(f"**Emotion:** {result['emotion']}")
                            st.write(f"**Emotion Confidence:** {result['emotion_confidence']:.1f}%")
                            
                            # Display individual face
                            st.image(result['face_roi'], caption=f"Face {i}", use_container_width=True)
                            st.markdown("---")
            else:
                st.warning("⚠️ No face detected in the image. Please try again.")

# ========== PAGE 4: STATISTICS ==========
elif page == "📊 Statistics":
    st.header("📊 Dataset Statistics")
    
    if not os.path.exists(DATASET_DIR):
        st.error("❌ No dataset found.")
    else:
        persons = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
        
        if not persons:
            st.warning("⚠️ No persons in dataset.")
        else:
            st.subheader("👥 Registered Persons")
            
            total_images = 0
            stats_data = []
            
            for person in persons:
                person_path = os.path.join(DATASET_DIR, person)
                images = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                image_count = len(images)
                total_images += image_count
                stats_data.append({"Person": person, "Images": image_count})
            
            st.dataframe(stats_data, use_container_width=True)
            st.metric("Total Persons", len(persons))
            st.metric("Total Images", total_images)
            
            # Check model status
            st.subheader("🤖 Model Status")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                emotion_exists = os.path.exists(EMOTION_MODEL_PATH)
                st.metric("Emotion Model", "✅ Loaded" if emotion_exists else "❌ Not Found")
            
            with col2:
                face_model_exists = os.path.exists(FACE_MODEL_PATH)
                st.metric("Face Model", "✅ Trained" if face_model_exists else "❌ Not Trained")
            
            with col3:
                labels_exists = os.path.exists(LABELS_PATH)
                st.metric("Labels", "✅ Available" if labels_exists else "❌ Not Available")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Face & Emotion Recognition System | Built with Streamlit</div>",
    unsafe_allow_html=True
)

