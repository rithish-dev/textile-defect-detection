from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO('results/fabric_defect_v1/weights/best.pt')

# Open webcam
cap = cv2.VideoCapture(0)

print("Running defect detection. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow('Fabric Defect Detector', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()