import os
import json
import cv2
import numpy as np
from pathlib import Path
from config import DATASET_DIR, MODEL_PATH, LABELS_PATH, FACE_SIZE

def ensure_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)

def build_dataset():
    labels_map = {}     # name -> id
    next_id = 0
    samples = []
    targets = []

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    dataset_root = Path(DATASET_DIR)

    for person_dir in dataset_root.iterdir():
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name
        if person_name not in labels_map:
            labels_map[person_name] = next_id
            next_id += 1

        for img_path in person_dir.glob("*.*"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
            if len(faces) == 0:
                continue

            # Use the largest detected face
            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, FACE_SIZE)

            samples.append(face_resized)
            targets.append(labels_map[person_name])

    return samples, targets, labels_map

def train_and_save(samples, targets, labels_map):
    if len(samples) == 0:
        raise RuntimeError("No faces found in dataset. Please add images to data\\dataset\\<PersonName>\\*.jpg")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(samples, np.array(targets))
    recognizer.write(MODEL_PATH)

    # Save id -> name mapping
    id_to_name = {str(v): k for k, v in labels_map.items()}
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(id_to_name, f, indent=2)
    print(f"Training complete. Model: {MODEL_PATH}, Labels: {LABELS_PATH}, Samples: {len(samples)}")

def main():
    ensure_dirs()
    samples, targets, labels_map = build_dataset()
    train_and_save(samples, targets, labels_map)

if __name__ == "__main__":
    main()