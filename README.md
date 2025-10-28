# Fall Detection System

This project implements a fall detection system using LSTM neural networks and pre-trained ResNet features.

## Project Structure

```
FallDetected/
├── load_data.py        # Stage 1: Load and extract features from videos
├── process_data.py     # Stage 2: Process data into sequences for training
├── train.py           # Stage 3: Train the LSTM model
├── main.py            # Main pipeline orchestrator
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Update the `DATA_DIR` path in `load_data.py` to point to your fall dataset directory.

## Usage

### Option 1: Run the complete pipeline
```bash
python main.py --stage all
```

### Option 2: Run individual stages

1. **Load Data** (Extract features from videos):
```bash
python main.py --stage load
```
or
```bash
python load_data.py
```

2. **Process Data** (Create sequences for training):
```bash
python main.py --stage process
```
or
```bash
python process_data.py
```

3. **Train Model** (Train the LSTM network):
```bash
python main.py --stage train
```
or
```bash
python train.py
```

### Evaluate and Validate trained models

**Basic Evaluation:**
```bash
python main.py --evaluate
```

**Comprehensive Validation:**
```bash
python main.py --validate
```

**Quick Validation:**
```bash
# Basic validation with key metrics
python quick_validate.py

# Comprehensive validation with detailed analysis
python quick_validate.py --full

# Validate specific model
python quick_validate.py --model fall_lstm_final.pth --full
```

**Direct Validation:**
```bash
python validate.py
```

## File Descriptions

### load_data.py
- Loads video files from the dataset
- Extracts features using pre-trained ResNet50
- Parses annotation files to get fall timestamps and bounding boxes
- Saves raw features and labels to numpy files

### process_data.py
- Loads raw features and labels
- Creates sequences of specified length for LSTM training
- Splits data into training and testing sets
- Saves processed data for training

### train.py
- Defines the LSTM model architecture
- Implements training loop with validation and early stopping
- Saves the best performing model during training
- Includes learning rate scheduling and gradient clipping
- Provides functions for model evaluation

### validate.py
- Comprehensive model validation suite
- Implements cross-validation, temporal validation
- Generates detailed metrics and visualizations
- Includes learning curve analysis and overfitting detection
- Saves validation results and plots

### quick_validate.py
- Quick validation script for immediate model assessment
- Supports both basic and comprehensive validation modes
- Easy-to-use command-line interface

### main.py
- Orchestrates the entire pipeline
- Provides command-line interface for running individual stages
- Handles error checking and progress reporting
- Includes validation options

## Model Architecture

The system uses a bidirectional LSTM network with the following architecture:
- Input: Sequences of ResNet50 features (2048-dimensional)
- LSTM: 2 layers, 128 hidden units, bidirectional
- Output: Binary classification (fall/no fall)

## Output Files

The pipeline generates several intermediate and final files:

**Data Files:**
- `raw_features.npy`: Raw features extracted from videos
- `raw_labels.npy`: Raw labels for each frame
- `X_train.npy`, `X_test.npy`: Training and testing feature sequences
- `y_train.npy`, `y_test.npy`: Training and testing labels

**Model Files:**
- `best_fall_lstm.pth`: Best performing model during training (includes training metadata)
- `fall_lstm_final.pth`: Final model state dict
- `training_history.pkl`: Training history (loss and accuracy curves)

**Validation Results:**
- `validation_results/`: Directory containing validation plots and metrics
  - `confusion_matrix.png`: Model confusion matrix
  - `roc_curve.png`: ROC curve with AUC score
  - `precision_recall_curve.png`: Precision-Recall curve
  - `learning_curves.png`: Training and validation learning curves
  - `cv_results.png`: Cross-validation results visualization
  - Detailed validation result files (`.pkl` format)

## Configuration

You can modify the following parameters in the respective files:
- Sequence length and step size in `process_data.py`
- Model architecture parameters in `train.py`
- Training hyperparameters (epochs, batch size, learning rate) in `train.py`
- Data paths and scene names in `load_data.py`

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- scikit-learn
- Pillow

## Notes

- The code is designed to work with GPU acceleration if available
- Make sure your dataset follows the expected directory structure
- The annotation files should contain fall timestamps and bounding box information
- Videos should be in .avi format (can be modified in the code)


đây là dòng sửa đổi
