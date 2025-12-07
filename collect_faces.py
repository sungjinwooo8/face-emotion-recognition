import cv2
import os

print(">>> collect_faces.py started")  # debug line

person_name = input("Enter your name (no spaces if possible): ").strip()
if not person_name:
    print("Name cannot be empty. Exiting.")
    exit()

DATASET_DIR = "faces_dataset"
person_dir = os.path.join(DATASET_DIR, person_name)
os.makedirs(person_dir, exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open webcam.")
    exit()

print(f"Capturing faces for: {person_name}")
print("Look at the camera. Press 'q' to stop early.")

count = 0
MAX_IMAGES = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (200, 200))

        img_path = os.path.join(person_dir, f"{person_name}_{count}.jpg")
        cv2.imwrite(img_path, face_roi)
        count += 1
        print(f"Saved image {count}/{MAX_IMAGES}", end="\r")

        if count >= MAX_IMAGES:
            break

    cv2.imshow("Collecting faces - " + person_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= MAX_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nDone! Saved {count} images to {person_dir}")
