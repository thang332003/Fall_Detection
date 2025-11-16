import os
import time
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort


class LSTMFallClassifier:
    """LSTM ONNXRuntime wrapper for fall/normal classification.

    Model has 2 output units:
      index 0 -> not fall
      index 1 -> fall

    This implementation assumes the LSTM was trained on feature vectors
    stored similarly to `raw_features.npy` with shape (N, F).
    It expects the ONNX model input shape (1, 1, F).
    """

    def __init__(
        self,
        model_path: str = "lstm.onnx",
        providers=None,
        feature_size: int = 128,
        not_fall_index: int = 0,
        fall_index: int = 1,
    ) -> None:
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.model_path = model_path
        self.feature_size = feature_size
        self.not_fall_index = not_fall_index
        self.fall_index = fall_index

        # Single reusable InferenceSession
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    # -------------------- Feature extraction --------------------
    def extract_features_from_roi(self, roi_bgr) -> np.ndarray:
        """Convert ROI to a feature vector for LSTM.

        Hiện tại training script của bạn dùng `raw_features.npy` (N, F).
        Nếu F == feature_size, thì mỗi mẫu là một vector F.

        Vì không có code trích xuất feature từ ảnh trong project gốc,
        ở đây tạm thời dùng cách đơn giản: resize ROI -> grayscale -> flatten,
        sau đó cắt/pad về `feature_size`. Điều này KHÔNG giống training,
        nhưng cho phép bạn chạy pipeline end-to-end.

        Khi bạn có cùng pipeline trích xuất feature như lúc train
        `fall_lstm_final.pth`, hãy thay phần này bằng đúng logic đó
        để kết quả chính xác.
        """
        roi_resized = cv2.resize(roi_bgr, (32, 32))
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        vec = gray.flatten().astype(np.float32) / 255.0

        if vec.shape[0] < self.feature_size:
            pad_len = self.feature_size - vec.shape[0]
            vec = np.pad(vec, (0, pad_len), mode="constant")
        elif vec.shape[0] > self.feature_size:
            vec = vec[: self.feature_size]

        return vec

    # -------------------- Inference --------------------
    def classify(self, roi_bgr) -> Tuple[str, float, float]:
        """Classify a single ROI as fall/not-fall.

        Returns:
            label: "fall" or "normal"
            confidence: float
            infer_time_ms: float
        """
        features = self.extract_features_from_roi(roi_bgr)

        # Example input shape: (batch, seq_len, feature_size) = (1, 1, F)
        inp = features.reshape(1, 1, -1).astype(np.float32)

        t0 = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: inp})[0]
        t1 = time.time()

        # Assume output shape: (1, num_classes) with softmax probabilities
        probs = outputs[0]
        not_fall_prob = float(probs[self.not_fall_index])
        fall_prob = float(probs[self.fall_index])

        if fall_prob >= not_fall_prob:
            label = "fall"
            conf = fall_prob
        else:
            label = "normal"
            conf = not_fall_prob

        infer_time_ms = (t1 - t0) * 1000.0
        return label, conf, infer_time_ms
