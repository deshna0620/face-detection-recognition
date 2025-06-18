from ultralytics import YOLO

model = YOLO("yolov8n.pt")

for img_path in ["data/face1.png", "data/face2.png", "data/face1_masked.png", "data/face2_capped.png"]:
    model.predict(
        source=img_path,
        conf=0.5,
        save=True,
        project="runs/detect",
        name="my_faces",
        exist_ok=True
    )

print("Face detection complete. Check 'runs/detect/my_faces' folder for output.")