# Basic configuration for the face attendance system

# Camera sources:
# - Use "0" for default webcam
# - Or set an RTSP URL string like: "rtsp://user:pass@192.168.1.10:554/stream"
CAMERA_SOURCES = [
    "0"
    # "rtsp://user:pass@192.168.1.10:554/stream"
]

# Paths (Windows-style backslashes)
DATASET_DIR = "data\\dataset"            # Folder containing subfolders per person
MODEL_PATH = "models\\lbph_model.xml"    # Trained model output
LABELS_PATH = "models\\labels.json"      # Mapping of label IDs to names
ATTENDANCE_CSV_PATH = "data\\attendance.csv"

# Face detection and recognition params
FACE_SIZE = (200, 200)                   # LBPH input size
RECOGNITION_CONFIDENCE_THRESHOLD = 65.0  # Lower is better; tune after training
MIN_FACE_SIZE = (80, 80)                 # Minimum face bounding box to consider

# Attendance logging params
DUPLICATE_COOLDOWN_SECONDS = 60          # Prevent duplicate logs for same person
SHOW_PREVIEW = True                      # Set to False for headless mode