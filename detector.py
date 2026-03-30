import os
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from ultralytics import YOLO
from transformers import SiglipForImageClassification, AutoImageProcessor
from ultralytics.nn.tasks import PoseModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules import Conv

torch.serialization.add_safe_globals([PoseModel, Sequential, Conv])


class FaceDetector:
    """YOLOv8 + ByteTrack face detector"""

    def __init__(self, model_path="yolov8n-face.pt", device="cpu"):
        print("[YOLOv8] Loading model...")
        self.model = YOLO(model_path)
        self.device = device
        print("[YOLOv8] Loaded")

    def detect_and_track(self, frame):
        """
        Returns: [(track_id, x, y, w, h), ...]
        """
        results = self.model.track(frame, persist=True, device=self.device, verbose=False)

        faces = []

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id

            if track_ids is None:
                continue

            track_ids = track_ids.cpu().numpy()

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = box

                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)

                faces.append((int(tid), x, y, w, h))

        return faces

    def extract_face(self, frame, x, y, w, h, padding=0.3):
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)

        return frame[y1:y2, x1:x2]


class SigLIPDetector:
    """Deepfake classifier"""

    MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"

    def __init__(self, device, conf_threshold=0.5):
        print("[SigLIP] Loading model...")
        self.device = device
        self.conf_threshold = conf_threshold

        self.model = SiglipForImageClassification.from_pretrained(self.MODEL_NAME)
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)

        self.model.to(device)
        self.model.eval()

        print("[SigLIP] Loaded")

    def predict(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return None

        # Resize face (standard for model)
        face_bgr = cv2.resize(face_bgr, (224, 224))

        # Convert to RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze()

        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

        # Confidence filtering
        if confidence < self.conf_threshold:
            return None

        label = "FAKE" if pred_idx == 0 else "REAL"

        return label, confidence