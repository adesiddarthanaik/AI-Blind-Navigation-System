"""
====================================================================
  AI-Powered Blind Navigation System — Main Entry Point
====================================================================

  Usage:
      python main.py                             # laptop webcam
      python main.py --source 1                  # external USB camera
      python main.py --source video.mp4          # video file
      python main.py --source http://ip:8080/video  # IP Webcam app
      python main.py --lang hi                   # Hindi audio
      python main.py --no-vlm                    # skip Moondream2
      python main.py --no-display                # headless (no window)

====================================================================
"""

import argparse
import time
import sys
import os
import cv2
import numpy as np

# Project modules
import config
from detector import PotholeDetector
from estimator import DistanceEstimator, PathSuggester
from alert_manager import AlertManager
from audio_engine import AudioEngine
from scene_describer import SceneDescriber
from annotator import annotate_frame


def parse_args():
    p = argparse.ArgumentParser(description="AI Blind Navigation System")
    p.add_argument("--source", default=str(config.CAMERA_SOURCE),
                   help="0=webcam, 1=USB cam, path=video, URL=IP camera")
    p.add_argument("--lang", default=config.AUDIO_LANG, choices=["en", "hi"],
                   help="Audio language")
    p.add_argument("--no-vlm", action="store_true",
                   help="Disable Moondream2 scene description")
    p.add_argument("--no-display", action="store_true",
                   help="No live video window")
    p.add_argument("--save", default=config.OUTPUT_VIDEO_PATH,
                   help="Output video path (set 'none' to disable)")
    return p.parse_args()


def open_camera(source):
    """Open camera, video file, or IP stream."""
    try:
        src = int(source)
    except ValueError:
        src = source

    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"\n[ERROR] Cannot open: {source}")
        print("  Options:")
        print("    python main.py --source 0              (webcam)")
        print("    python main.py --source video.mp4      (video file)")
        print("    python main.py --source http://ip:8080/video  (IP Webcam)")
        sys.exit(1)

    return cap


def main():
    args = parse_args()

    # ---- Banner ----
    print("\n" + "=" * 60)
    print("  AI-POWERED BLIND NAVIGATION SYSTEM")
    print("=" * 60)
    print(f"  Source    : {args.source}")
    print(f"  Language  : {args.lang}")
    print(f"  VLM       : {'ON (int8)' if not args.no_vlm else 'OFF'}")
    print(f"  Display   : {'ON' if not args.no_display else 'OFF'}")
    print(f"  Save to   : {args.save}")
    print("=" * 60)

    # ============================================================
    # 1. LOAD YOLO
    # ============================================================
    print("\n[1/5] Loading YOLO detector...")
    detector = PotholeDetector(
        weights_path=config.YOLO_WEIGHTS,
        confidence=config.YOLO_CONFIDENCE,
        iou=config.YOLO_IOU,
    )
    detector.load()

    # ============================================================
    # 2. INIT ESTIMATOR + PATH
    # ============================================================
    print("\n[2/5] Initializing estimator...")
    estimator = DistanceEstimator(
        calibration_y=config.CALIBRATION_Y,
        calibration_d=config.CALIBRATION_D,
        fov_h=config.CAMERA_FOV_HORIZONTAL,
        stride_m=config.STRIDE_M,
    )
    path_suggester = PathSuggester()

    # ============================================================
    # 3. LOAD MOONDREAM2 (int8)
    # ============================================================
    print("\n[3/5] Loading scene describer...")
    vlm_enabled = config.ENABLE_VLM and (not args.no_vlm)
    scene = SceneDescriber(
        model_id=config.VLM_MODEL_ID,
        revision=config.VLM_REVISION,
        prompt=config.SCENE_PROMPT,
        interval_sec=config.VLM_INTERVAL_SEC,
        enabled=vlm_enabled,
    )
    scene.load()

    # ============================================================
    # 4. INIT ALERT MANAGER
    # ============================================================
    print("\n[4/5] Initializing alert manager...")
    alert_mgr = AlertManager(
        lang=args.lang,
        dist_critical=config.DIST_CRITICAL,
        dist_warning=config.DIST_WARNING,
        cooldown_critical=config.COOLDOWN_CRITICAL_SEC,
        cooldown_warning=config.COOLDOWN_WARNING_SEC,
        cooldown_info=config.COOLDOWN_INFO_SEC,
        pixel_thresh=config.COOLDOWN_PIXEL_THRESH,
        scene_interval=config.VLM_INTERVAL_SEC,
        scene_quiet_gap=config.VLM_QUIET_GAP_SEC,
    )

    # ============================================================
    # 5. START AUDIO
    # ============================================================
    print("\n[5/5] Starting audio engine...")
    audio = AudioEngine(lang=args.lang, rate=config.AUDIO_RATE)
    audio.start()

    # ============================================================
    # OPEN CAMERA
    # ============================================================
    print(f"\nOpening camera: {args.source}")
    cap = open_camera(args.source)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tw = config.FRAME_WIDTH
    th = config.FRAME_HEIGHT

    alert_mgr.set_fps(fps)

    print(f"  Source res  : {src_w}x{src_h}")
    print(f"  Process res : {tw}x{th}")
    print(f"  FPS         : {fps}")
    if total_frames > 0:
        print(f"  Frames      : {total_frames}")
        print(f"  Duration    : {total_frames / fps:.1f} sec")

    # ---- Output video writer ----
    writer = None
    if args.save.lower() != "none":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (tw, th))
        print(f"  Saving to   : {args.save}")

    # ---- Scene interval in frames ----
    scene_interval = int(config.VLM_INTERVAL_SEC * fps)

    # ============================================================
    # MAIN LOOP
    # ============================================================
    print("\n" + "=" * 60)
    print("  RUNNING — Press 'q' to quit")
    print("=" * 60 + "\n")

    frame_num = 0
    total_dets = 0
    t_start = time.time()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                if total_frames > 0:
                    print("\n[INFO] Video ended.")
                else:
                    print("\n[WARN] Camera disconnected.")
                break

            # Resize
            frame_bgr = cv2.resize(frame_bgr, (tw, th))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            canvas = frame_rgb.copy()

            # ---- YOLO ----
            raw_dets = detector.detect(frame_rgb)

            # ---- Estimate ----
            results = []
            for d in raw_dets:
                info = estimator.estimate(
                    d["x1"], d["y1"], d["x2"], d["y2"], th, tw
                )
                info["class_name"] = d["class_name"]
                info["confidence"] = d["confidence"]
                info["bbox"] = [int(d["x1"]), int(d["y1"]),
                                int(d["x2"]), int(d["y2"])]
                results.append(info)
                total_dets += 1

            # ---- Path ----
            path_text = path_suggester.suggest(results)

            # ---- Scene (every N frames) ----
            scene_text = None
            if scene.is_loaded() and frame_num % scene_interval == 0:
                scene_text = scene.describe(frame_rgb, force=True)

            # ---- Alert Manager ----
            alert = alert_mgr.process_frame(
                detections=results,
                frame_num=frame_num,
                scene_text=scene_text,
                path_text=path_text,
            )

            # ---- Audio ----
            if alert["speak_text"]:
                pri_map = {"critical": 3, "warning": 2, "info": 1, "scene": 0}
                audio.speak(
                    alert["speak_text"],
                    priority=pri_map.get(alert["alert_type"], 1),
                )

            # ---- Annotate ----
            canvas = annotate_frame(
                canvas, results, th, tw,
                scene_text=alert["display_scene"],
                path_text=alert["display_path"],
                dist_critical=config.DIST_CRITICAL,
                dist_warning=config.DIST_WARNING,
            )

            # ---- Write output ----
            if writer:
                writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

            # ---- Display ----
            if not args.no_display:
                cv2.imshow(config.WINDOW_NAME, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[INFO] 'q' pressed. Stopping.")
                    break

            # ---- Progress ----
            frame_num += 1
            if frame_num % (fps * 2) == 0:
                elapsed = time.time() - t_start
                speed = frame_num / elapsed if elapsed > 0 else 0
                if total_frames > 0:
                    pct = frame_num / total_frames * 100
                    eta = (total_frames - frame_num) / speed if speed > 0 else 0
                    msg = (f"  Frame {frame_num}/{total_frames} ({pct:.0f}%) | "
                           f"{speed:.1f} fps | ETA: {eta:.0f}s | Dets: {total_dets}")
                else:
                    msg = f"  Frame {frame_num} | {speed:.1f} fps | Dets: {total_dets}"
                print(msg, end="\r")

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted.")

    finally:
        # ---- Cleanup ----
        elapsed = time.time() - t_start
        cap.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        audio.stop()

        # ---- Summary ----
        print("\n" + "=" * 60)
        print("  SESSION SUMMARY")
        print("=" * 60)
        print(f"  Frames       : {frame_num}")
        print(f"  Time         : {elapsed:.1f} sec")
        if elapsed > 0:
            print(f"  Avg FPS      : {frame_num / elapsed:.1f}")
        print(f"  Detections   : {total_dets}")

        log = alert_mgr.get_log()
        print(f"  Alerts       : {len(log)}")

        if log:
            print(f"\n  Last 15 Alerts:")
            for e in log[-15:]:
                print(f"    [{e['time_sec']:6.1f}s] [{e['type']:8s}] {e['text'][:70]}")

        if writer and args.save.lower() != "none":
            print(f"\n  Output: {args.save}")

        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
