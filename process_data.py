import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

def prepare_sequences(features_list, labels_list, seq_length=20, step=10):
    """Prepare sequences with correct 3D shape and print data/label counts."""
    sequence_features = []
    sequence_labels = []
    
    for feats, frame_labels in zip(features_list, labels_list):
        if len(feats) < seq_length:
            continue
        for i in range(0, len(feats) - seq_length + 1, step):
            seq = feats[i:i+seq_length]  # Shape [seq_length, 2048]
            seq_labels = frame_labels[i:i+seq_length]
            label = 1 if np.any(seq_labels == 1) else 0
            sequence_features.append(seq)
            sequence_labels.append(label)
    
    print(f"Number of data sequences prepared: {len(sequence_features)}")
    print(f"Number of labels prepared: {len(sequence_labels)}")
    print(f"Shape of features_list: {np.array(sequence_features).shape if sequence_features else 'N/A'}")
    return np.array(sequence_features), np.array(sequence_labels)

class FallDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Shape [num_samples, seq_length, input_size]
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def process_data(seq_length=20, step=10, test_size=0.2, random_state=42):
    """Process raw data into training and testing datasets."""
    print("Loading raw data...")
    try:
        raw_features = np.load('raw_features.npy', allow_pickle=True)
        raw_labels = np.load('raw_labels.npy', allow_pickle=True)
    except FileNotFoundError:
        print("Raw data files not found. Please run load_data.py first.")
        return None, None, None, None, None, None
    
    print("Preparing sequences...")
    X, y = prepare_sequences(raw_features, raw_labels, seq_length, step)
    
    if len(X) == 0:
        print("No data loaded! Check folder names and paths.")
        return None, None, None, None, None, None
    
    print(f"X shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Create datasets
    train_dataset = FallDataset(X_train, y_train)
    test_dataset = FallDataset(X_test, y_test)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Fall sequences in training: {np.sum(y_train)}")
    print(f"Normal sequences in training: {len(y_train) - np.sum(y_train)}")
    print(f"Fall sequences in testing: {np.sum(y_test)}")
    print(f"Normal sequences in testing: {len(y_test) - np.sum(y_test)}")
    
    return X_train, X_test, y_train, y_test, train_dataset, test_dataset

if __name__ == "__main__":
    print("Processing data...")
    X_train, X_test, y_train, y_test, train_dataset, test_dataset = process_data()
    
    if X_train is not None:
        # Save processed data
        np.save('X_train.npy', X_train)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)
        print("Processed data saved to X_train.npy, X_test.npy, y_train.npy, y_test.npy")
    else:
        print("Data processing failed.")
