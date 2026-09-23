import cv2
import time
from datetime import datetime

STREAM_URL = "rtsp://127.0.0.1:8080/h264.sdp"

camera = cv2.VideoCapture(STREAM_URL)

if not camera.isOpened():
    print("❌ Could not connect to phone camera.")
    exit()

print("✅ Phone camera connected!")
print("Motion detector started.")
print("Press Q to quit.")

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

while True:

    ret, frame2 = camera.read()

    if not ret:
        print("❌ Lost camera stream.")
        break

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

        with open("motion.log", "a") as log:
            log.write(message + "\n")

        last_motion_time = current_time

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

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

print("Motion detector stopped.")
print(f"Total motion events: {motion_count}")
