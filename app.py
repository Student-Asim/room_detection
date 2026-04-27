from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

model_path = r"C:\Users\sarua\Desktop\room_detection_system\runs\detect\room_damage_yolov8s-8\weights\best.pt"
model = YOLO(model_path)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class_map = {
    0: "door_window",
    1: "fire_damage",
    2: "paint_damage",
    3: "crack_damage",
}

# Custom confidence thresholds for different damage types
# This allows for stricter detection on ambiguous damage
CLASS_THRESHOLDS = {
    "door_window": 0.25,  # Usually easier to detect
    "fire_damage": 0.25,  # Requires more confidence
    "paint_damage": 0.25,
    "crack_damage": 0.25  # Often small, needs more confidence
}

def process_image(img):
    """Core function with TTA and Custom Thresholding."""
    # augment=True enables Test Time Augmentation (TTA)
    # TTA increases accuracy by predicting on multiple versions of the image
    results = model.predict(
        img, 
        conf=0.25, # Base threshold for YOLO, we filter stricter below
        iou=0.45, 
        imgsz=640, 
        device='cpu', 
        verbose=False,
        augment=True 
    )[0]
    
    counts = {name: 0 for name in class_map.values()}
    detections = []

    for box in results.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = class_map.get(cls_id, "unknown")
        
        # Apply Class-Specific Thresholding
        required_conf = CLASS_THRESHOLDS.get(class_name, 0.45)
        
        if conf >= required_conf:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            counts[class_name] = counts.get(class_name, 0) + 1
            detections.append({
                "class": class_name,
                "confidence": round(conf, 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            })
        
    return {"counts": counts, "detections": detections}

@app.post("/detect")
async def detect_room_elements(image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return process_image(img)

@app.post("/detect_in_batch")
async def detect_in_batch(files: List[UploadFile] = File(...)):
    batch_results = {}
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        batch_results[file.filename] = process_image(img)
    return batch_results