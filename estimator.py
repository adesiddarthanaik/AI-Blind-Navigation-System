"""
Estimator — Distance, Angle, Steps, Path Suggestion.
Pure geometry. No ML model needed.

Math:
    bbox bottom-y   → calibrated polynomial → DISTANCE (meters)
    bbox center-x   → pinhole geometry      → ANGLE (degrees)
    bbox width       → perspective formula   → REAL WIDTH (meters)
    distance/stride  → division              → STEPS TO REACH
    width/stride     → division              → STEPS TO CROSS
"""

import math
import numpy as np


class DistanceEstimator:

    def __init__(self, calibration_y, calibration_d, fov_h=60, stride_m=0.50):
        """
        Args:
            calibration_y: list of normalized bbox bottom-y values
            calibration_d: list of real distances in meters
            fov_h: camera horizontal FOV in degrees
            stride_m: walking stride in meters
        """
        self.fov_h = fov_h
        self.stride = stride_m

        if len(calibration_y) < 3 or len(calibration_d) < 3:
            raise ValueError("Need at least 3 calibration points.")
        if len(calibration_y) != len(calibration_d):
            raise ValueError("calibration_y and calibration_d must be same length.")

        coeffs = np.polyfit(calibration_y, calibration_d, deg=2)
        self.poly_fn = np.poly1d(coeffs)
        print(f"[Estimator] Ready | FOV: {fov_h}° | Stride: {stride_m}m")

    def estimate(self, x1, y1, x2, y2, frame_h, frame_w):
        """
        Full estimation from one bounding box.

        Returns dict with:
            distance_m, angle_deg, direction_text,
            real_width_m, steps_to_reach, steps_to_cross
        """
        # DISTANCE: bottom-y position → polynomial → meters
        bottom_y_norm = y2 / frame_h
        distance_m = float(self.poly_fn(bottom_y_norm))
        distance_m = round(max(0.3, min(distance_m, 15.0)), 2)

        # ANGLE: center-x → pinhole geometry → degrees
        center_x = (x1 + x2) / 2.0
        angle_deg = round((center_x / frame_w - 0.5) * self.fov_h, 1)

        # DIRECTION: angle → human-readable text
        direction_text = self._angle_to_text(angle_deg)

        # REAL WIDTH: perspective formula
        fov_rad = math.radians(self.fov_h)
        bbox_w_ratio = (x2 - x1) / frame_w
        real_width_m = round(
            bbox_w_ratio * distance_m * 2 * math.tan(fov_rad / 2), 2
        )

        # STEPS
        steps_to_reach = max(1, round(distance_m / self.stride))
        steps_to_cross = max(1, math.ceil(real_width_m / self.stride))

        return {
            "distance_m": distance_m,
            "angle_deg": angle_deg,
            "direction_text": direction_text,
            "real_width_m": real_width_m,
            "steps_to_reach": steps_to_reach,
            "steps_to_cross": steps_to_cross,
        }

    def _angle_to_text(self, angle_deg):
        a = abs(angle_deg)
        side = "left" if angle_deg < 0 else "right"
        if a < 5:
            return "directly ahead"
        elif a < 15:
            return f"slightly to your {side}"
        elif a < 25:
            return f"to your {side}"
        else:
            return f"far {side}"


class PathSuggester:
    """
    Divides frame into 3 corridors by angle.
    Counts obstacles per corridor.
    Suggests safest direction.
    """

    def suggest(self, detections):
        if not detections:
            return "Path is clear ahead"

        zones = {"left": 0, "center": 0, "right": 0}
        for det in detections:
            angle = det["angle_deg"]
            if angle < -10:
                zones["left"] += 1
            elif angle > 10:
                zones["right"] += 1
            else:
                zones["center"] += 1

        safest = min(zones, key=zones.get)
        if zones[safest] == 0:
            return f"Move {safest}, path is clear there"
        else:
            return "Multiple obstacles ahead, slow down carefully"
