"""
Annotator — Draws visual annotations on frames.

Color-coded by urgency:
    RED    = DANGER  (< 2m)
    ORANGE = WARNING (2-5m)
    GREEN  = SAFE    (> 5m)

Each detection gets:
    - Bounding box with corner accents
    - Info panel: zone, class, distance, angle, direction, steps
    - Center badge: steps + angle
    - Scene bar (top)
    - Path suggestion bar (bottom)
"""

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate_frame(canvas, detections, frame_h, frame_w,
                   scene_text="", path_text="",
                   dist_critical=2.0, dist_warning=5.0):
    """
    Draw all annotations on canvas (modified in-place).

    Args:
        canvas: numpy (H, W, 3) RGB
        detections: list of dicts from estimator
        frame_h, frame_w: frame dimensions
        scene_text: VLM scene description
        path_text: path suggestion
        dist_critical: meters for DANGER
        dist_warning: meters for WARNING

    Returns:
        annotated canvas
    """

    for det in detections:
        bx1, by1, bx2, by2 = det["bbox"]
        conf = det["confidence"]
        cls = det["class_name"]
        dist = det["distance_m"]
        angle = det["angle_deg"]
        dirn = det["direction_text"]
        reach = det["steps_to_reach"]
        cross = det["steps_to_cross"]

        # ---- Color ----
        if dist < dist_critical:
            color, zone = (255, 50, 50), "DANGER"
        elif dist < dist_warning:
            color, zone = (255, 180, 0), "WARNING"
        else:
            color, zone = (50, 200, 50), "SAFE"

        # ---- Bounding box ----
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 3)

        # ---- Corner accents ----
        cl = min(20, (bx2 - bx1) // 3, (by2 - by1) // 3)
        if cl > 3:
            for cx, cy, dx, dy in [
                (bx1, by1, 1, 1), (bx2, by1, -1, 1),
                (bx1, by2, 1, -1), (bx2, by2, -1, -1),
            ]:
                cv2.line(canvas, (cx, cy), (cx + dx * cl, cy), color, 5)
                cv2.line(canvas, (cx, cy), (cx, cy + dy * cl), color, 5)

        # ---- Angle display ----
        if angle < -1:
            ang_str = f"{abs(angle)}\u00b0 L"
        elif angle > 1:
            ang_str = f"{abs(angle)}\u00b0 R"
        else:
            ang_str = "0\u00b0"

        # ---- Info panel above bbox ----
        fs, ft, lh, pad = 0.45, 1, 18, 5
        lines = [
            f"{zone} | {cls} ({conf:.0%})",
            f"Dist: {dist}m | Angle: {ang_str}",
            f"Dir: {dirn}",
            f"Reach: {reach} steps | Cross: {cross} steps",
        ]

        widths = [cv2.getTextSize(l, FONT, fs, ft)[0][0] for l in lines]
        pw = max(widths) + pad * 2
        ph = len(lines) * lh + pad * 2
        py1 = max(by1 - ph - 4, 0)
        py2 = py1 + ph
        px1 = bx1
        px2 = min(px1 + pw, frame_w)

        # Dark semi-transparent background
        ov = canvas.copy()
        cv2.rectangle(ov, (px1, py1), (px2, py2), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.65, canvas, 0.35, 0, canvas)

        for j, line in enumerate(lines):
            ty = py1 + pad + (j + 1) * lh - 3
            cv2.putText(canvas, line, (px1 + pad, ty),
                        FONT, fs, color, ft, cv2.LINE_AA)

        # ---- Center badge ----
        bcx = (bx1 + bx2) // 2
        bcy = (by1 + by2) // 2
        badge = f"{reach} steps | {ang_str}"
        (tw, th), _ = cv2.getTextSize(badge, FONT, 0.50, 2)

        cv2.rectangle(canvas,
                      (bcx - tw // 2 - 6, bcy - th // 2 - 6),
                      (bcx + tw // 2 + 6, bcy + th // 2 + 6),
                      color, -1)
        cv2.putText(canvas, badge,
                    (bcx - tw // 2, bcy + th // 2),
                    FONT, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

    # ---- Scene bar (top) ----
    if scene_text:
        ov = canvas.copy()
        cv2.rectangle(ov, (0, 0), (frame_w, 30), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.7, canvas, 0.3, 0, canvas)
        cv2.putText(canvas, f"Scene: {scene_text[:100]}",
                    (10, 20), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # ---- Path bar (bottom) ----
    if path_text:
        ov = canvas.copy()
        by = frame_h - 30
        cv2.rectangle(ov, (0, by), (frame_w, frame_h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.7, canvas, 0.3, 0, canvas)
        cv2.putText(canvas, f">> {path_text}",
                    (10, frame_h - 10), FONT, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

    return canvas
