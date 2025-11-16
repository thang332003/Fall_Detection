import numpy as np
import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    """LSTM model definition used only for ONNX export.

    This does NOT depend on raw_features.npy anymore. We fix input_size
    to a constant (must match feature_size in APP/model_lstm.py).
    """

    def __init__(self, input_size: int = 128, hidden_size: int = 64, num_layers: int = 1, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out


def main():
    """Export fall_lstm_final.pth (if available) to APP/lstm.onnx.

    - Không còn dùng raw_features.npy
    - Đặt cứng feature_size = 128
    - Nếu load được fall_lstm_final.pth thì dùng weight, không thì dùng random
    """
    feature_size = 128  # phải KHỚP với feature_size trong APP/model_lstm.py
    print("Using fixed feature_size for LSTM:", feature_size)

    model = SimpleLSTM(input_size=feature_size)

    # Load trained weights if available
    try:
        state = torch.load("fall_lstm_final.pth", map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print("Loaded weights from fall_lstm_final.pth")
    except Exception as e:  # noqa: BLE001
        print("Không load được fall_lstm_final.pth, dùng weight random:", e)

    model.eval()

    # Dummy input: (batch=1, seq_len=1, feature_size)
    dummy = torch.randn(1, 1, feature_size, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        "APP/lstm.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        dynamic_axes=None,
    )

    print("Exported LSTM to APP/lstm.onnx with feature_size =", feature_size)


if __name__ == "__main__":
    main()
