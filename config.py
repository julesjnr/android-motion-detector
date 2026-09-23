"""Shared configuration for the phone camera scripts.

Reads settings from environment variables so the stream URL doesn't
need to be hardcoded/duplicated across scripts.

    STREAM_URL   - RTSP URL of the phone camera (default below)
    HEADLESS     - if set to "1"/"true", skip cv2.imshow (no display needed)
"""

import os

STREAM_URL = os.environ.get("STREAM_URL", "rtsp://127.0.0.1:8080/h264.sdp")
HEADLESS = os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes")

# FRAME_SKIP: only run the (expensive) processing/detection logic on every
# Nth frame. Frames are still read every loop to keep the stream buffer
# from backing up, but skipped frames bypass grayscale/blur/diff/contours.
# Useful on low-powered hardware (e.g. Raspberry Pi) where full-rate
# processing can't keep up with the incoming stream.
FRAME_SKIP = max(1, int(os.environ.get("FRAME_SKIP", "1")))

# TARGET_FPS: optional cap on how many frames per second get processed.
# Leave unset (0) to process as fast as the stream/hardware allows.
TARGET_FPS = float(os.environ.get("TARGET_FPS", "0"))


def open_camera(source, retries=5, retry_delay=2):
    """Open a cv2.VideoCapture with a few retries, since phone RTSP
    streams (esp. over Wi-Fi) commonly fail to connect on the first try.
    """
    import time
    import cv2

    for attempt in range(1, retries + 1):
        camera = cv2.VideoCapture(source)
        if camera.isOpened():
            return camera
        print(f"⚠️  Connection attempt {attempt}/{retries} failed, retrying in {retry_delay}s...")
        camera.release()
        time.sleep(retry_delay)
    return None


class FrameRateLimiter:
    """Throttles a loop to at most TARGET_FPS iterations/sec, if set.
    Call .wait() once per loop iteration, after reading a frame.
    A no-op when TARGET_FPS is 0 (unset).
    """

    def __init__(self, target_fps):
        import time as _time
        self._time = _time
        self.min_interval = (1.0 / target_fps) if target_fps > 0 else 0
        self._last = self._time.time()

    def wait(self):
        if self.min_interval <= 0:
            return
        now = self._time.time()
        elapsed = now - self._last
        remaining = self.min_interval - elapsed
        if remaining > 0:
            self._time.sleep(remaining)
        self._last = self._time.time()
