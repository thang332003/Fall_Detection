from typing import Generator

import cv2
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from model_yolo import YoloPersonDetector
from model_lstm import LSTMFallClassifier

app = FastAPI()

# Mount static/templates if needed (here only templates via simple HTMLResponse)

# Preload models (same as app.py)
yolo_detector = YoloPersonDetector(model_path="yolo.onnx", providers=["CPUExecutionProvider"], input_size=(640, 640), conf_threshold=0.4)
lstm_classifier = LSTMFallClassifier(model_path="lstm.onnx", providers=["CPUExecutionProvider"], feature_size=128)

CONF_MIN_FALL = 0.5  # threshold to consider a detection as real fall


@app.get("/", response_class=HTMLResponse)
async def index():
    """Simple page that shows the MJPEG video stream."""
    # You can also move this HTML into templates/index.html if you want
    html = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <title>Fall Detection - YOLO + LSTM</title>
  <style>
    body { font-family: Arial, sans-serif; background: #222; color: #eee; text-align: center; }
    #video { border: 2px solid #555; margin-top: 20px; max-width: 90vw; }
    h1 { margin-top: 20px; }
  </style>
</head>
<body>
  <h1>Fall Detection - YOLO + LSTM (Webcam)</h1>
  <p>Using yolo.onnx to detect people and lstm.onnx to classify falls.</p>
  <img id=\"video\" src=\"/video_feed\" alt=\"Video stream\" />
</body>
</html>"""
    return HTMLResponse(content=html)


def gen_frames() -> Generator[bytes, None, None]:
    """Capture frames from webcam, run YOLO+LSTM, and yield MJPEG frames.

    Chỉ vẽ 1 bounding box trên mỗi frame: ưu tiên người có xác suất fall cao nhất.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera 0")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, _ = yolo_detector.detect_persons(frame)

        best_info = None  # (x, y, w, h, label, conf)

        for (x, y, w, h, score) in boxes:
            roi = frame[y : y + h, x : x + w]
            if roi.size == 0:
                continue

            label, conf, _ = lstm_classifier.classify(roi)

            # Ưu tiên fall có conf cao nhất, nếu không có fall thì lấy conf cao nhất bất kỳ
            if best_info is None:
                best_info = (x, y, w, h, label, conf)
            else:
                _, _, _, _, best_label, best_conf = best_info
                if label == "fall" and conf >= best_conf:
                    best_info = (x, y, w, h, label, conf)
                elif best_label != "fall" and conf > best_conf:
                    # chưa có fall nào, chọn conf cao hơn
                    best_info = (x, y, w, h, label, conf)

        # Sau khi duyệt hết, chỉ vẽ 1 box tốt nhất (nếu có)
        if best_info is not None:
            x, y, w, h, label, conf = best_info

            color = (0, 255, 0)
            text = f"normal {conf:.2f}"

            if label == "fall" and conf >= CONF_MIN_FALL:
                color = (0, 0, 255)
                text = f"fall {conf:.2f}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                text,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()


@app.get("/video_feed")
async def video_feed():
    """MJPEG video stream endpoint."""
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
