"""
Detector — YOLOv5 pothole detection wrapper.
Includes PosixPath fix for models trained on Linux/Colab.
"""

import os
import sys
import pathlib
import torch

# ============================================================
# FIX: PosixPath error when loading Linux-trained model on Windows
# This MUST be before torch.hub.load or torch.load
# ============================================================
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath


class PotholeDetector:

    def __init__(self, weights_path, confidence=0.45, iou=0.50):
        self.weights_path = weights_path
        self.confidence = confidence
        self.iou = iou
        self.model = None
        self.class_names = {}

    def load(self):
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Weights not found: {self.weights_path}\n"
                f"Place your best.pt in the project directory."
            )

        print(f"[Detector] Loading YOLOv5 from {self.weights_path}...")

        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=self.weights_path,
            force_reload=True,
        )
        self.model.conf = self.confidence
        self.model.iou = self.iou
        self.class_names = self.model.names

        device = next(self.model.model.parameters()).device
        print(f"[Detector] Loaded | Classes: {self.class_names} | Device: {device}")

    def detect(self, frame_rgb):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = self.model(frame_rgb)
        raw = results.xyxy[0].cpu().numpy()

        detections = []
        for det in raw:
            x1, y1, x2, y2, conf, cls_id = det
            detections.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": round(float(conf), 3),
                "class_name": self.class_names[int(cls_id)],
                "class_id": int(cls_id),
            })

        return detections