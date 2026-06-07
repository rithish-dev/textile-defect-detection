from ultralytics import YOLO

# Load YOLOv8 nano model (smallest and fastest)
model = YOLO('yolov8n.pt')

# Train on fabric defect dataset
results = model.train(
    data='data/fabric.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='fabric_defect_v1',
    project='results'
)

print("Training complete. Model saved in results/fabric_defect_v1/") 
