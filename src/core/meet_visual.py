import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

@dataclass
class VisualSpeakerEvent:
    timestamp: float  # seconds since recording start
    speaker_name: str  # OCR'd name from Meet tile

class MeetVisualDetector:
    """Detects active speaker in Google Meet via screenshots + color/OCR analysis."""

    def __init__(self):
        self._running = False
        self._thread = None
        self._events: list[VisualSpeakerEvent] = []
        self._lock = threading.Lock()
        self._start_time = None

    def start(self, start_time: float):
        """Start periodic screenshot capture."""
        self._start_time = start_time
        self._running = True
        self._events = []
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[VisualSpeakerEvent]:
        """Stop capture and return collected events."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        return list(self._events)

    def _capture_loop(self):
        """Capture screenshots at ~4 FPS and detect active speaker."""
        try:
            import mss
            import cv2
            import numpy as np
        except ImportError:
            log.warning("mss/opencv nao instalado, deteccao visual desativada")
            return

        with mss.mss() as sct:
            while self._running:
                try:
                    # Find Google Meet window
                    # Capture full screen (Meet window detection could be added later)
                    screenshot = sct.grab(sct.monitors[1])  # Primary monitor
                    img = np.array(screenshot)

                    # Convert BGRA to BGR
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                    # Detect active speaker border (Meet's blue-green highlight)
                    # HSV range for Meet's active speaker border
                    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

                    # Meet active speaker border: blue-green, approx HSV(90-115, 150-255, 150-255)
                    lower = np.array([90, 150, 150])
                    upper = np.array([115, 255, 255])
                    mask = cv2.inRange(hsv, lower, upper)

                    # Find contours of the highlighted border
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        # Get the largest contour (likely the active speaker tile border)
                        largest = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(largest)

                        if area > 500:  # Minimum area threshold
                            x, y, w, h = cv2.boundingRect(largest)

                            # OCR the name label at bottom of the tile
                            # Name is typically in the bottom 15% of the tile
                            name_region_y = y + int(h * 0.85)
                            name_region = img_bgr[name_region_y:y+h, x:x+w]

                            speaker_name = self._ocr_name(name_region)
                            if speaker_name:
                                elapsed = time.time() - self._start_time
                                with self._lock:
                                    self._events.append(VisualSpeakerEvent(
                                        timestamp=elapsed,
                                        speaker_name=speaker_name,
                                    ))

                    time.sleep(0.25)  # ~4 FPS

                except Exception as e:
                    log.debug(f"Erro na captura visual: {e}")
                    time.sleep(1)

    def _ocr_name(self, image) -> str | None:
        """Extract name text from image region using pytesseract."""
        try:
            import pytesseract
            import cv2

            # Preprocess: grayscale, threshold for white text on dark bg
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

            text = pytesseract.image_to_string(thresh, config='--psm 7').strip()

            # Clean up OCR result
            text = re.sub(r'[^\w\s]', '', text).strip()

            if len(text) >= 2:
                return text
        except Exception:
            pass
        return None

    def get_speaker_at_time(self, timestamp: float, window: float = 2.0) -> str | None:
        """Find the visual speaker closest to a given timestamp."""
        with self._lock:
            best = None
            best_diff = float('inf')
            for event in self._events:
                diff = abs(event.timestamp - timestamp)
                if diff < best_diff and diff <= window:
                    best_diff = diff
                    best = event.speaker_name
            return best
