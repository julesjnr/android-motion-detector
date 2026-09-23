import cv2
import time

STREAM_URL = "rtsp://127.0.0.1:8080/h264.sdp"

print("Connecting to Android phone camera...")

camera = cv2.VideoCapture(STREAM_URL)

if not camera.isOpened():
    print("❌ Could not connect to phone camera.")
    exit()

print("✅ Android camera connected!")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("❌ Could not receive frame.")
        break

    cv2.imshow("Phone Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Camera stopped.")
