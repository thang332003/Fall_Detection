from ultralytics import YOLO
import os
import shutil


def main():
    # Load YOLOv8n PyTorch model
    model = YOLO("yolov8n.pt")

    # Export to ONNX with current Ultralytics API
    # This returns the path to the exported ONNX file
    onnx_path = model.export(
        format="onnx",
        opset=12,
        imgsz=640,
        dynamic=False,
    )

    print("Raw ONNX exported to:", onnx_path)

    # Ensure APP directory exists
    app_dir = os.path.join(os.path.dirname(__file__), "APP")
    os.makedirs(app_dir, exist_ok=True)

    # Copy to APP/yolo.onnx so app.py can load it
    target_path = os.path.join(app_dir, "yolo.onnx")
    shutil.copy2(onnx_path, target_path)
    print("Copied ONNX to:", target_path)


if __name__ == "__main__":
    main()