import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from process_data import FallDataset

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available, using CPU")

class FallLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=128, num_layers=2):
        super(FallLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        out = self.fc(h_n)
        return self.sigmoid(out)

def train_model(epochs=100, batch_size=32, learning_rate=0.001, num_workers=2, 
               use_preprocessing=True, early_stopping=True, patience=15, 
               validation_split=0.15):
    """Train the fall detection model with validation and early stopping."""
    print("Loading processed data...")
    try:
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
        print("✓ Loaded existing processed data")
    except FileNotFoundError:
        print("Processed data files not found. Running data processing...")
        from process_data import process_data
        X_train, X_test, y_train, y_test, _, _ = process_data()
        if X_train is None:
            print("Failed to process data.")
            return None
        print("✓ Data processed successfully")
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Fall/Normal ratio in training: {np.sum(y_train)}/{len(y_train) - np.sum(y_train)}")
    
    # Split training data into train/validation
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=validation_split, random_state=42, stratify=y_train
    )
    
    print(f"After validation split:")
    print(f"  Train: {len(X_train_split)} samples")
    print(f"  Validation: {len(X_val_split)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Create datasets and data loaders
    train_dataset = FallDataset(X_train_split, y_train_split)
    val_dataset = FallDataset(X_val_split, y_val_split)
    test_dataset = FallDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            num_workers=num_workers, pin_memory=True)
    
    # Initialize model, loss, and optimizer
    model = FallLSTM().to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    if early_stopping:
        print(f"Early stopping enabled with patience: {patience}")
    
    # Training tracking
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Store history
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['train_acc'].append(train_accuracy)
        training_history['val_acc'].append(val_accuracy)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
            print(f'  Val Loss:   {avg_val_loss:.4f}, Val Acc:   {val_accuracy:.4f}')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'training_history': training_history
            }, 'best_fall_lstm.pth')
            print(f'  ✓ New best model saved! Val Acc: {best_val_accuracy:.4f}')
        else:
            epochs_without_improvement += 1
        
        # Early stopping check
        if early_stopping and epochs_without_improvement >= patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            print(f'Best validation accuracy: {best_val_accuracy:.4f}')
            break
    
    # Final evaluation on test set
    print(f'\nTraining completed!')
    print(f'Best validation accuracy: {best_val_accuracy:.4f}')
    
    # Load best model for final evaluation
    checkpoint = torch.load('best_fall_lstm.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_accuracy = evaluate_model(model, test_loader)
    print(f'Final test accuracy: {test_accuracy:.2f}%')
    
    # Save final model state (just the state dict for compatibility)
    torch.save(model.state_dict(), 'fall_lstm_final.pth')
    print("Final model saved as fall_lstm_final.pth")
    
    # Save training history
    import pickle
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(training_history, f)
    print("Training history saved as training_history.pkl")
    
    return model

def evaluate_model(model, test_loader):
    """Evaluate the model and return accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            outputs = model(batch_X)
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def load_and_evaluate(model_path='best_fall_lstm.pth'):
    """Load a saved model and evaluate it."""
    print(f"Loading model from {model_path}...")
    try:
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
    except FileNotFoundError:
        print("Test data files not found. Please run process_data.py first.")
        return
    
    test_dataset = FallDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=2, pin_memory=True)
    
    model = FallLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    accuracy = evaluate_model(model, test_loader)
    print(f'Model accuracy: {accuracy:.2f}%')
    
    return model

if __name__ == "__main__":
    # Train the model
    model = train_model(epochs=100, batch_size=32, learning_rate=0.001)
    
    if model is not None:
        print("\nTraining completed successfully!")
        print("Models saved:")
        print("- best_fall_lstm.pth (best performing model during training)")
        print("- fall_lstm_final.pth (final model after all epochs)")
    else:
        print("Training failed. Please check the data files.")
