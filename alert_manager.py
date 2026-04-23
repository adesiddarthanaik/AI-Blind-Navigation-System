"""
Alert Manager — Coordinates all alert systems.

Priority: CRITICAL > WARNING > INFO > SCENE

Handles:
    - Cooldown (prevents repeated alerts for same pothole)
    - Alert merging (multiple potholes → one clean sentence)
    - Scene description timing (fills gaps, never interrupts)
    - Path suggestion (speaks only when changed)
"""

import math


class AlertManager:

    def __init__(self, lang="en", dist_critical=2.0, dist_warning=5.0,
                 cooldown_critical=1.0, cooldown_warning=2.0, cooldown_info=3.0,
                 pixel_thresh=50, scene_interval=4.0, scene_quiet_gap=2.0):

        self.lang = lang
        self.fps = 30

        # Thresholds
        self.dist_critical = dist_critical
        self.dist_warning = dist_warning

        # Cooldown in seconds per level
        self._cooldown_sec = {3: cooldown_critical, 2: cooldown_warning, 1: cooldown_info}
        self._pixel_thresh = pixel_thresh

        # Scene timing in seconds
        self._scene_interval = scene_interval
        self._scene_quiet_gap = scene_quiet_gap

        # State
        self._active_zones = []
        self._last_alert_frame = -999
        self._last_scene_frame = -999
        self._last_path_text = ""
        self._last_scene_text = ""
        self._frames_since_danger = 999
        self._log = []

    def set_fps(self, fps):
        self.fps = max(1, fps)

    def reset(self):
        self._active_zones = []
        self._last_alert_frame = -999
        self._last_scene_frame = -999
        self._last_path_text = ""
        self._last_scene_text = ""
        self._frames_since_danger = 999
        self._log = []

    # ---- Urgency ----

    def _level(self, dist):
        if dist < self.dist_critical:
            return 3
        elif dist < self.dist_warning:
            return 2
        return 1

    def _level_tag(self, lv):
        return {3: "DANGER", 2: "WARNING", 1: "SAFE"}.get(lv, "")

    # ---- Cooldown ----

    def _cooldown_frames(self, level):
        return int(self._cooldown_sec.get(level, 2.0) * self.fps)

    def _cleanup(self, frame_num):
        self._active_zones = [
            (cx, cy, fn, lv)
            for cx, cy, fn, lv in self._active_zones
            if (frame_num - fn) < self._cooldown_frames(lv)
        ]

    def _is_duplicate(self, cx, cy, frame_num):
        self._cleanup(frame_num)
        for ax, ay, _, _ in self._active_zones:
            if math.sqrt((cx - ax) ** 2 + (cy - ay) ** 2) < self._pixel_thresh:
                return True
        return False

    def _register(self, cx, cy, frame_num, level):
        self._active_zones.append((cx, cy, frame_num, level))

    # ---- Text building ----

    def _single_alert(self, det, level):
        name = det["class_name"]
        reach = det["steps_to_reach"]
        cross = det["steps_to_cross"]
        dirn = det["direction_text"]
        angle = abs(det["angle_deg"])

        if self.lang == "hi":
            if dirn == "directly ahead":
                d = "bilkul saamne"
            else:
                side = "baayein" if det["angle_deg"] < 0 else "daayein"
                d = f"{int(angle)} degree {side}"
            if level == 3:
                return f"KHATARAA! {name} {reach} kadam aage, {d}!"
            return f"{name}, {reach} kadam aage, {d}, {cross} kadam chauda"
        else:
            if dirn == "directly ahead":
                d = "directly ahead"
            else:
                d = f"{dirn}, about {int(angle)} degrees"
            if level == 3:
                return f"DANGER! {name} {reach} steps ahead, {d}!"
            return f"{name}, {reach} steps ahead, {d}, {cross} steps wide"

    def _merged_alert(self, dets):
        sorted_d = sorted(dets, key=lambda x: x["steps_to_reach"])
        parts = []
        if len(sorted_d) > 2:
            n = len(sorted_d)
            parts.append(f"{n} potholes ahead" if self.lang == "en" else f"{n} potholes saamne")
        for det in sorted_d[:3]:
            lv = self._level(det["distance_m"])
            parts.append(self._single_alert(det, lv))
        return ". ".join(parts)

    # ---- Main method ----

    def process_frame(self, detections, frame_num, scene_text=None, path_text=None):
        """
        Process all detections for one frame.

        Returns dict:
            speak_text    : what to speak (or None)
            alert_type    : 'critical'/'warning'/'info'/'scene'/None
            display_scene : latest scene text for overlay
            display_path  : latest path text for overlay
        """

        # Step 1: Filter through cooldown
        new_dets = []
        for det in detections:
            bx1, by1, bx2, by2 = det["bbox"]
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2
            lv = self._level(det["distance_m"])

            if not self._is_duplicate(cx, cy, frame_num):
                self._register(cx, cy, frame_num, lv)
                det["_level"] = lv
                new_dets.append(det)

        # Step 2: Max urgency
        max_lv = max((d["_level"] for d in new_dets), default=0)

        if max_lv >= 2:
            self._frames_since_danger = 0
        else:
            self._frames_since_danger += 1

        # Step 3: Build alert by priority
        speak = None
        atype = None

        if max_lv == 3:
            speak = self._merged_alert(new_dets)
            atype = "critical"
            self._last_alert_frame = frame_num

        elif max_lv == 2:
            if (frame_num - self._last_alert_frame) >= self.fps * 1.5:
                speak = self._merged_alert(new_dets)
                atype = "warning"
                self._last_alert_frame = frame_num

        elif max_lv == 1:
            if (frame_num - self._last_alert_frame) >= self.fps * 3:
                speak = self._merged_alert(new_dets)
                atype = "info"
                self._last_alert_frame = frame_num

        # Step 4: Path suggestion (only if changed)
        if path_text:
            if speak and path_text != self._last_path_text:
                speak += f". {path_text}"
                self._last_path_text = path_text

        # Step 5: Scene description (only in quiet periods)
        if scene_text and scene_text != self._last_scene_text:
            self._last_scene_text = scene_text

            no_alert = (speak is None)
            quiet = self._frames_since_danger > (self._scene_quiet_gap * self.fps)
            gap_ok = (frame_num - self._last_scene_frame) > (self._scene_interval * self.fps)

            if no_alert and quiet and gap_ok:
                speak = f"Scene: {scene_text}"
                atype = atype or "scene"
                self._last_scene_frame = frame_num

        # Log
        if speak:
            self._log.append({
                "frame": frame_num,
                "time_sec": round(frame_num / self.fps, 1),
                "type": atype,
                "text": speak,
            })

        return {
            "speak_text": speak,
            "alert_type": atype,
            "display_scene": self._last_scene_text,
            "display_path": self._last_path_text or (path_text or ""),
        }

    def get_log(self):
        return self._log
