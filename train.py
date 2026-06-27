from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11s.pt")

    model.train(
        data="Hard-Hat-Universe-17/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
    )
