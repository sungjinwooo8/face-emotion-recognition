import cv2
import numpy as np
import tensorflow as tf
import pickle

# ---------- 1. Load models ----------

EMOTION_MODEL_PATH = "emotion_model_IIITM.h5"
emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)

FACE_MODEL_PATH = "face_recognizer.yml"
LABELS_PATH = "labels.pickle"

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(FACE_MODEL_PATH)

with open(LABELS_PATH, "rb") as f:
    label_ids = pickle.load(f)

id_to_name = {v: k for k, v in label_ids.items()}

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# class order from your Colab model:
# {'neutral':0, 'sad':1, 'smile':2, 'surprise':3, 'surprise_open':4, 'yawning':5}
EMOTIONS = ['neutral', 'sad', 'smile', 'surprise', 'surprise_open', 'yawning']
IMG_SIZE = 48

def preprocess_for_emotion(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    x = gray.astype("float32") / 255.0
    x = np.expand_dims(x, axis=-1)
    x = np.expand_dims(x, axis=0)
    return x

# ---------- 2. Webcam loop ----------

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: cannot open webcam")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rects = face_cascade.detectMultiScale(
        gray_full, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces_rects:
        face_roi_color = frame[y:y + h, x:x + w]
        face_roi_gray = gray_full[y:y + h, x:x + w]

        # ---- Face recognition ----
        face_gray_resized = cv2.resize(face_roi_gray, (200, 200))
        label_id, conf = recognizer.predict(face_gray_resized)
        # lower conf = better match
        if conf < 90:
            person_name = id_to_name.get(label_id, "Unknown")
        else:
            person_name = "Unknown"

        # ---- Emotion prediction ----
        emo_input = preprocess_for_emotion(face_roi_color)
        emo_probs = emotion_model.predict(emo_input, verbose=0)[0]
        emo_idx = int(np.argmax(emo_probs))
        emo_label = EMOTIONS[emo_idx]
        emo_conf = emo_probs[emo_idx]

        text = f"{person_name} | {emo_label} ({emo_conf:.2f})"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face + Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
