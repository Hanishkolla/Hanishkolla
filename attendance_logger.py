import csv
import os
from pathlib import Path
from datetime import datetime

def ensure_csv(path):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "person", "camera"])

def log_attendance(path, person, camera):
    ensure_csv(path)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ts, person, camera])