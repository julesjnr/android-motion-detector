import cv2

from config import STREAM_URL, HEADLESS, TARGET_FPS, open_camera, FrameRateLimiter

RECONNECT_RETRIES = 5
RECONNECT_DELAY = 2  # seconds

print("Connecting to Android phone camera...")

camera = open_camera(STREAM_URL, retries=RECONNECT_RETRIES, retry_delay=RECONNECT_DELAY)

if camera is None:
    print("❌ Could not connect to phone camera after retries.")
    exit()

print("✅ Android camera connected!")
if not HEADLESS:
    print("Press Q to quit.")
else:
    print("Running headless. Press Ctrl+C to quit.")

rate_limiter = FrameRateLimiter(TARGET_FPS)

try:
    while True:
        success, frame = camera.read()
        rate_limiter.wait()

        if not success:
            print("⚠️  Lost frame, attempting to reconnect...")
            camera.release()
            camera = open_camera(STREAM_URL, retries=RECONNECT_RETRIES, retry_delay=RECONNECT_DELAY)
            if camera is None:
                print("❌ Could not reconnect. Stopping.")
                break
            print("✅ Reconnected!")
            continue

        if not HEADLESS:
            cv2.imshow("Phone Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("\nStopped by user (Ctrl+C).")

camera.release()
if not HEADLESS:
    cv2.destroyAllWindows()

print("Camera stopped.")
