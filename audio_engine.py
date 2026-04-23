"""
Audio Engine — Non-blocking threaded TTS using pyttsx3.

Runs in a background thread so detection never pauses.
Priority queue: CRITICAL alerts jump ahead of queued ones.
"""

import threading
import queue
import time


class AudioEngine:

    def __init__(self, lang="en", rate=175):
        self.lang = lang
        self.rate = rate
        self._queue = queue.PriorityQueue()
        self._thread = None
        self._running = False
        self._log = []

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("[Audio] Started (background thread)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        print(f"[Audio] Stopped | Alerts spoken: {len(self._log)}")

    def speak(self, text, priority=1):
        """
        Queue text for speaking.
        priority: 3=critical, 2=warning, 1=info, 0=scene
        """
        if text and self._running:
            # Negate so higher priority (3) gets picked first from min-heap
            self._queue.put((-priority, time.time(), text))

    def get_log(self):
        return self._log

    def _worker(self):
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)

            # Try to set Hindi voice if available
            if self.lang == "hi":
                for v in engine.getProperty("voices"):
                    if "hindi" in v.name.lower() or "hi" in v.id.lower():
                        engine.setProperty("voice", v.id)
                        break

            print(f"[Audio] pyttsx3 initialized (lang={self.lang}, rate={self.rate})")

        except ImportError:
            print("[Audio] pyttsx3 not found. Install: pip install pyttsx3")
            print("[Audio] Running in SILENT mode (alerts logged but not spoken)")
            engine = None

        except Exception as e:
            print(f"[Audio] pyttsx3 init failed: {e}")
            print("[Audio] Running in SILENT mode")
            engine = None

        while self._running:
            try:
                neg_pri, ts, text = self._queue.get(timeout=0.5)
                self._log.append({
                    "time": ts,
                    "priority": -neg_pri,
                    "text": text,
                })

                if engine:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as e:
                        print(f"[Audio] Speak error: {e}")
                else:
                    print(f"[Audio] (silent) {text[:80]}")

                self._queue.task_done()

            except queue.Empty:
                continue

        if engine:
            try:
                engine.stop()
            except Exception:
                pass
