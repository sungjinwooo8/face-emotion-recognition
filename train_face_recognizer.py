import cv2
import os
import numpy as np
import pickle

DATASET_DIR = "faces_dataset"

faces = []
labels = []
label_ids = {}
current_id = 0

for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

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

faces = np.array(faces)
labels = np.array(labels)

print("Total faces:", len(faces))
print("Label mapping:", label_ids)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

recognizer.write("face_recognizer.yml")
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

print("Saved face_recognizer.yml and labels.pickle")
