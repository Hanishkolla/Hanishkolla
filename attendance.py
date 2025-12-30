import cv2
import json
import time
from pathlib import Path
from config import (
    CAMERA_SOURCES, MODEL_PATH, LABELS_PATH, ATTENDANCE_CSV_PATH,
    FACE_SIZE, RECOGNITION_CONFIDENCE_THRESHOLD, MIN_FACE_SIZE,
    DUPLICATE_COOLDOWN_SECONDS, SHOW_PREVIEW
)
from utils.attendance_logger import log_attendance

def load_model_and_labels():
    # Load LBPH model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run training first.")
    recognizer.read(MODEL_PATH)

    # Load labels
    if not Path(LABELS_PATH).exists():
        raise FileNotFoundError(f"Labels not found at {LABELS_PATH}. Please run training first.")
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)  # id(str) -> name
    labels = {int(k): v for k, v in labels.items()}
    return recognizer, labels

def open_capture(source):
    # source can be "0" (string) for webcam or RTSP URL
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera source: {source}")
    return cap

def recognize_and_log():
    recognizer, labels = load_model_and_labels()
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    last_seen = {}  # person_name -> last timestamp logged

    for idx, source in enumerate(CAMERA_SOURCES):
        # One camera at a time (simple version); for multi-camera, spawn threads
        cap = open_capture(source)
        camera_name = f"Camera_{idx}"

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.5)
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=MIN_FACE_SIZE)

                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, FACE_SIZE)

                    label_id, confidence = recognizer.predict(face_resized)
                    person_name = labels.get(label_id, "Unknown")

                    is_known = person_name != "Unknown" and confidence <= RECOGNITION_CONFIDENCE_THRESHOLD
                    color = (0, 255, 0) if is_known else (0, 0, 255)

                    if is_known:
                        now = time.time()
                        last = last_seen.get(person_name, 0)
                        if now - last >= DUPLICATE_COOLDOWN_SECONDS:
                            log_attendance(ATTENDANCE_CSV_PATH, person_name, camera_name)
                            last_seen[person_name] = now

                    if SHOW_PREVIEW:
                        text = f"{person_name}" if is_known else "Unknown"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, f"{text} ({confidence:.1f})", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if SHOW_PREVIEW:
                    cv2.imshow(camera_name, frame)
                    # Press 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            if SHOW_PREVIEW:
                cv2.destroyAllWindows()

def main():
    recognize_and_log()

if __name__ == "__main__":
    main()