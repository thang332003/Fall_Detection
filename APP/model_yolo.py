import time
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort


class YoloPersonDetector:
    """YOLOv8 ONNXRuntime wrapper for person detection (class id 0).

    Assumes `yolo.onnx` is an export of `yolov8n.pt` with ONNX output shape:
        (1, 84, 8400)
    where for each prediction index i (0..8399):
        - outputs[0, 0, i] = x_center
        - outputs[0, 1, i] = y_center
        - outputs[0, 2, i] = width
        - outputs[0, 3, i] = height
        - outputs[0, 4:, i] = class scores (80 classes, COCO)

    We filter for class 0 (person) and above confidence threshold.
    """

    def __init__(
        self,
        model_path: str = "yolo.onnx",
        providers=None,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        person_class_id: int = 0,
    ) -> None:
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.person_class_id = person_class_id

        print(f"[YOLO] Loading ONNX model from {self.model_path} with providers={providers}")
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    # -------------------- Pre / Post processing --------------------
    def _letterbox(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize image with unchanged aspect ratio using padding (letterbox)."""
        h, w = image.shape[:2]
        new_w, new_h = self.input_size

        scale = min(new_w / w, new_h / h)
        resized_w, resized_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (resized_w, resized_h))

        canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        pad_x = (new_w - resized_w) // 2
        pad_y = (new_h - resized_h) // 2
        canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized

        return canvas, scale, (pad_x, pad_y)

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Prepare frame for YOLO model: BGR->RGB, resize, normalize, CHW, batch."""
        img, scale, pad = self._letterbox(frame_bgr)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        img_rgb = np.transpose(img_rgb, (2, 0, 1))  # HWC -> CHW
        img_rgb = np.expand_dims(img_rgb, axis=0)    # NCHW
        return img_rgb, scale, pad

    def _postprocess(
        self,
        outputs: np.ndarray,
        scale: float,
        pad: Tuple[int, int],
        orig_shape: Tuple[int, int],
    ) -> List[Tuple[int, int, int, int, float]]:
        """Convert YOLOv8 ONNX (1, 84, 8400) to person bounding boxes."""
        h0, w0 = orig_shape
        pad_x, pad_y = pad

        # outputs: (1, 84, 8400)
        preds = outputs[0]  # (84, 8400)
        num_preds = preds.shape[1]

        # x, y, w, h: (8400,)
        x = preds[0]
        y = preds[1]
        w = preds[2]
        h = preds[3]

        # class scores: (80, 8400)
        cls_scores = preds[4:]
        cls_ids = np.argmax(cls_scores, axis=0)  # (8400,)
        scores = cls_scores[cls_ids, np.arange(num_preds)]  # (8400,)

        boxes: List[Tuple[int, int, int, int, float]] = []

        for i in range(num_preds):
            score = float(scores[i])
            cls_id = int(cls_ids[i])

            if score < self.conf_threshold:
                continue
            if cls_id != self.person_class_id:
                continue

            x_c = float(x[i])
            y_c = float(y[i])
            bw = float(w[i])
            bh = float(h[i])

            # Undo letterbox padding and scaling (x,y,w,h ở toạ độ letterboxed)
            x_c -= pad_x
            y_c -= pad_y
            x_c /= scale
            y_c /= scale
            bw /= scale
            bh /= scale

            x1 = max(int(x_c - bw / 2), 0)
            y1 = max(int(y_c - bh / 2), 0)
            x2 = min(int(x_c + bw / 2), w0 - 1)
            y2 = min(int(y_c + bh / 2), h0 - 1)

            bw_i = max(x2 - x1, 0)
            bh_i = max(y2 - y1, 0)
            if bw_i <= 0 or bh_i <= 0:
                continue

            boxes.append((x1, y1, bw_i, bh_i, score))

        print(f"[YOLO] Postprocess found {len(boxes)} person boxes (conf>={self.conf_threshold})")
        return boxes

    # -------------------- Public API --------------------
    def detect_persons(self, frame_bgr: np.ndarray):
        """Run person detection on a single BGR frame."""
        orig_h, orig_w = frame_bgr.shape[:2]
        inp, scale, pad = self._preprocess(frame_bgr)

        t0 = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: inp})[0]
        t1 = time.time()
        print(f"[YOLO] ONNX raw output shape: {outputs.shape}")
        boxes = self._postprocess(outputs, scale, pad, (orig_h, orig_w))
        infer_time_ms = (t1 - t0) * 1000.0
        return boxes, infer_time_ms
