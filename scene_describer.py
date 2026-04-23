"""
Scene Describer — Moondream2 VLM (int8 quantized).

Quantized to 8-bit: ~2GB VRAM instead of ~3.8GB.
Fits alongside YOLOv5s (~0.5GB) on a 4GB GPU.

Gives the blind person scene awareness beyond pothole detection:
vehicles, animals, road condition, construction, etc.
"""

import time
import torch
from PIL import Image


class SceneDescriber:

    def __init__(self, model_id, revision, prompt, interval_sec=4.0, enabled=True):
        """
        Args:
            model_id    : HuggingFace model ID
            revision    : model revision/tag
            prompt      : scene description prompt
            interval_sec: seconds between descriptions
            enabled     : set False to skip VLM entirely
        """
        self.model_id = model_id
        self.revision = revision
        self.prompt = prompt
        self.interval = interval_sec
        self.enabled = enabled

        self.model = None
        self.tokenizer = None
        self._last_run = 0
        self._last_text = ""
        self._loaded = False

    def load(self):
        """Load Moondream2 in int8 quantized mode."""
        if not self.enabled:
            print("[Scene] VLM disabled in config.")
            return False

        if not torch.cuda.is_available():
            print("[Scene] No CUDA GPU. VLM requires GPU. Disabling.")
            self.enabled = False
            return False

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"[Scene] Loading {self.model_id} (int8 quantized)...")
            print(f"[Scene] Revision: {self.revision}")
            print(f"[Scene] Expected VRAM: ~2.0 GB")
            print(f"[Scene] First run downloads ~3.7 GB model files\n")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=True,
                attn_implementation="eager",
                load_in_8bit=True,       # INT8 QUANTIZATION
                device_map="auto",       # auto GPU/CPU split
            )

            self.model.eval()

            params = sum(p.numel() for p in self.model.parameters()) / 1e9
            print(f"[Scene] Loaded (int8) | Params: ~{params:.1f}B")

            # Print VRAM usage
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"[Scene] GPU VRAM used: {allocated:.2f} GB")

            self._loaded = True
            return True

        except ImportError as e:
            if "bitsandbytes" in str(e):
                print("[Scene] ERROR: bitsandbytes not installed.")
                print("[Scene] Run: pip install bitsandbytes")
                print("[Scene] Then restart and try again.")
            else:
                print(f"[Scene] Import error: {e}")
            self._loaded = False
            return False

        except torch.cuda.OutOfMemoryError:
            print("[Scene] ERROR: GPU out of memory.")
            print("[Scene] Your GPU cannot fit Moondream2 even in int8.")
            print("[Scene] Run with --no-vlm to disable scene descriptions.")
            self._loaded = False
            return False

        except Exception as e:
            print(f"[Scene] Failed to load: {e}")
            self._loaded = False
            return False

    def describe(self, frame_rgb, force=False):
        """
        Generate scene description from a frame.

        Args:
            frame_rgb: numpy array (H, W, 3) RGB
            force: ignore interval timer if True

        Returns:
            string description or None
        """
        if not self._loaded or self.model is None:
            return None

        now = time.time()
        if not force and (now - self._last_run) < self.interval:
            return None

        self._last_run = now

        try:
            pil_image = Image.fromarray(frame_rgb)

            with torch.no_grad():
                enc = self.model.encode_image(pil_image)
                answer = self.model.answer_question(
                    enc, self.prompt, self.tokenizer
                )

            self._last_text = answer.strip()
            return self._last_text

        except torch.cuda.OutOfMemoryError:
            print("[Scene] GPU OOM during inference. Disabling VLM.")
            self._loaded = False
            return None

        except Exception as e:
            print(f"[Scene] Inference error: {e}")
            return None

    def get_last(self):
        return self._last_text

    def is_loaded(self):
        return self._loaded
