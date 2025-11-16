"""External AI detector for Shinobi CCTV (WebSocket-based).

This server:
- Receives frames from Shinobi via WebSocket as JSON with base64 JPEG or buffer
- Decodes to OpenCV BGR frame
- Runs YOLO (ONNX) to detect persons
- For each person ROI, runs LSTM (ONNX) to classify fall vs normal
- Returns JSON in Shinobi Detector format:

    {
      "objects": [
        {
          "tag": "fall",
          "confidence": 0.91,
          "x": 120,
          "y": 200,
          "w": 180,
          "h": 260
        }
      ]
    }

Run:
    pip install onnxruntime opencv-python websockets numpy
    python app.py
"""

import asyncio
import base64
import json
import logging
import os
import signal
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import websockets

from model_yolo import YoloPersonDetector
from model_lstm import LSTMFallClassifier

# -------------------- Logging setup --------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("shinobi_fall_detector")

# -------------------- Models (preloaded, reused 24/7) --------------------
yolo_detector = YoloPersonDetector(
    model_path="yolo.onnx",
    providers=["CPUExecutionProvider"],  # change to ["CUDAExecutionProvider", "CPUExecutionProvider"] if GPU
    input_size=(640, 640),
    conf_threshold=0.4,
)

lstm_classifier = LSTMFallClassifier(
    model_path="lstm.onnx",
    providers=["CPUExecutionProvider"],
    feature_size=128,
)


# -------------------- Frame decoding helpers --------------------
def decode_frame_from_message(message: Dict[str, Any]) -> np.ndarray:
    """Decode a frame sent by Shinobi.

    Expected JSON example from Shinobi (you can adapt keys if needed):
        {
          "imageBase64": "<base64_jpeg>"
        }

    Or raw buffer base64 under "buffer" key.
    """
    if "imageBase64" in message:
        b64_str = message["imageBase64"]
        img_bytes = base64.b64decode(b64_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame

    if "buffer" in message:
        b64_str = message["buffer"]
        img_bytes = base64.b64decode(b64_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame

    raise ValueError("Message does not contain 'imageBase64' or 'buffer'.")


def build_shinobi_response(objects: List[Dict[str, Any]]) -> str:
    """Build JSON string in Shinobi Detector format."""
    return json.dumps({"objects": objects}, ensure_ascii=False)


async def process_frame(message: Dict[str, Any]) -> str:
    """Full pipeline for a single frame.

    - Decode frame
    - YOLO person detection
    - LSTM fall classification on each person ROI
    - Build JSON response
    """
    t0 = time.time()

    frame = decode_frame_from_message(message)
    if frame is None:
        logger.warning("Decoded frame is None")
        return build_shinobi_response([])

    # 1) YOLO detection
    boxes, t_yolo = yolo_detector.detect_persons(frame)
    logger.info(f"YOLO inference time: {t_yolo:.2f} ms, persons detected: {len(boxes)}")

    objects: List[Dict[str, Any]] = []

    # 2) For each detected person -> LSTM fall classification
    for (x, y, w, h, score) in boxes:
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        label, conf, t_lstm = lstm_classifier.classify(roi)
        logger.info(
            "LSTM inference: %.2f ms, label=%s, conf=%.3f, bbox=(%d,%d,%d,%d), yolo_score=%.3f",
            t_lstm,
            label,
            conf,
            x,
            y,
            w,
            h,
            score,
        )

        # Only report fall to Shinobi, or include normal if desired
        if label == "fall":
            objects.append(
                {
                    "tag": label,
                    "confidence": float(conf),
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                }
            )

    t1 = time.time()
    logger.info(
        "Total frame processing: %.2f ms, objects returned: %d",
        (t1 - t0) * 1000.0,
        len(objects),
    )

    return build_shinobi_response(objects)


# -------------------- WebSocket Server --------------------
async def websocket_handler(websocket, path):
    """Handle a WebSocket connection from Shinobi.

    For each incoming frame/message, run the AI pipeline and send back JSON.
    """
    logger.info("New WebSocket connection from %s", websocket.remote_address)

    try:
        async for raw_msg in websocket:
            try:
                if isinstance(raw_msg, bytes):
                    raw_msg = raw_msg.decode("utf-8", errors="ignore")
                message = json.loads(raw_msg)

                response = await process_frame(message)
                await websocket.send(response)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error while processing message: %s", exc)
                # In case of error, respond with empty objects list
                await websocket.send(build_shinobi_response([]))
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in websocket handler: %s", exc)


async def main_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start WebSocket server and run forever."""
    logger.info("Starting WebSocket server on ws://%s:%d", host, port)
    async with websockets.serve(
        websocket_handler,
        host,
        port,
        max_size=10 * 1024 * 1024,  # up to ~10MB per frame
    ):
        await asyncio.Future()  # run forever


def run() -> None:
    """Entry point for running 24/7 with graceful shutdown."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Graceful shutdown on Ctrl+C
    def _stop_loop(*_args: Any) -> None:
        logger.info("Received stop signal, shutting down event loop...")
        loop.stop()

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is not None:
            try:
                loop.add_signal_handler(sig, _stop_loop)
            except (NotImplementedError, RuntimeError):
                # Signal handling may not be available on some platforms
                pass

    try:
        loop.run_until_complete(main_server())
    finally:
        loop.close()
        logger.info("Server stopped.")


if __name__ == "__main__":
    run()
