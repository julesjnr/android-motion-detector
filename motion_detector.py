import time
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

import cv2

from config import STREAM_URL, HEADLESS, FRAME_SKIP, TARGET_FPS, open_camera, FrameRateLimiter

# --- Logging setup (rotates at 1MB, keeps 3 backups instead of growing forever) ---
logger = logging.getLogger("motion")
logger.setLevel(logging.INFO)
_handler = RotatingFileHandler("motion.log", maxBytes=1_000_000, backupCount=3)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)

RECONNECT_RETRIES = 5
RECONNECT_DELAY = 2  # seconds between reconnect attempts

camera = open_camera(STREAM_URL, retries=RECONNECT_RETRIES, retry_delay=RECONNECT_DELAY)

if camera is None:
    print("❌ Could not connect to phone camera after retries.")
    exit()

print("✅ Phone camera connected!")
print("Motion detector started.")
if not HEADLESS:
    print("Press Q to quit.")
else:
    print("Running headless. Press Ctrl+C to quit.")


def read_frame_with_reconnect(camera, source):
    """Try to read a frame; if the stream dropped, attempt to reconnect
    a few times before giving up, instead of exiting immediately.
    """
    ret, frame = camera.read()
    if ret:
        return camera, ret, frame

    print("⚠️  Lost camera stream, attempting to reconnect...")
    camera.release()
    camera = open_camera(source, retries=RECONNECT_RETRIES, retry_delay=RECONNECT_DELAY)
    if camera is None:
        return None, False, None

    print("✅ Reconnected!")
    ret, frame = camera.read()
    return camera, ret, frame


# Read the first frame
ret, frame1 = camera.read()

if not ret:
    print("❌ Could not read first frame.")
    camera.release()
    exit()

# Convert first frame to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

last_motion_time = 0
motion_cooldown = 2  # seconds

motion_count = 0
frame_number = 0
rate_limiter = FrameRateLimiter(TARGET_FPS)

try:
    while True:

        camera, ret, frame2 = read_frame_with_reconnect(camera, STREAM_URL)

        if not ret or camera is None:
            print("❌ Lost camera stream and could not reconnect.")
            break

        frame_number += 1
        rate_limiter.wait()

        # Skip the expensive detection work on non-sampled frames.
        # Still shows the live feed (if not headless) so the video doesn't
        # look frozen, but no diff/contour work is done on skipped frames.
        if frame_number % FRAME_SKIP != 0:
            if not HEADLESS:
                cv2.imshow("Android Motion Detector", frame2)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        # Convert current frame to grayscale
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

        # Compare the two frames
        difference = cv2.absdiff(gray1, gray2)

        # Detect areas that changed
        _, threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

        # Expand detected areas
        threshold = cv2.dilate(threshold, None, iterations=2)

        # Find moving objects
        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False

        for contour in contours:

            area = cv2.contourArea(contour)

            # Ignore tiny changes
            if area < 1000:
                continue

            motion_detected = True

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                frame2,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Log motion
        current_time = time.time()

        if motion_detected and current_time - last_motion_time > motion_cooldown:

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            motion_count += 1

            message = f"{timestamp} - Motion detected #{motion_count}"

            print(message)
            logger.info(message)

            last_motion_time = current_time

        if not HEADLESS:
            # Display status
            if motion_detected:
                cv2.putText(
                    frame2,
                    "MOTION DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
            else:
                cv2.putText(
                    frame2,
                    "NO MOTION",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            # Show camera
            cv2.imshow("Android Motion Detector", frame2)

        # Previous frame becomes current frame
        gray1 = gray2

        if not HEADLESS:
            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("\nStopped by user (Ctrl+C).")

camera.release()
if not HEADLESS:
    cv2.destroyAllWindows()

print("Motion detector stopped.")
print(f"Total motion events: {motion_count}")
