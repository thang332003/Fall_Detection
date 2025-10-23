import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
import os
import pickle
from datetime import datetime
from process_data import FallDataset
from train import FallLSTM

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ModelValidator:
    """Comprehensive validation class for fall detection model."""
    
    def __init__(self, model_path='best_fall_lstm.pth', save_plots=True, results_dir='validation_results'):
        self.model_path = model_path
        self.save_plots = save_plots
        self.results_dir = results_dir
        self.create_results_dir()
        
    def create_results_dir(self):
        """Create directory for saving validation results."""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"Created results directory: {self.results_dir}")
    
    def load_data(self):
        """Load processed data for validation."""
        try:
            print("Loading processed data...")
            X_train = np.load('X_train.npy')
            X_test = np.load('X_test.npy')
            y_train = np.load('y_train.npy')
            y_test = np.load('y_test.npy')
            
            print(f"Training data: {X_train.shape}, Labels: {y_train.shape}")
            print(f"Testing data: {X_test.shape}, Labels: {y_test.shape}")
            print(f"Fall/Normal ratio in test: {np.sum(y_test)}/{len(y_test) - np.sum(y_test)}")
            
            return X_train, X_test, y_train, y_test
            
        except FileNotFoundError:
            print("Processed data files not found. Please run process_data.py first.")
            return None, None, None, None
    
    def load_model(self, model_path=None):
        """Load trained model."""
        if model_path is None:
            model_path = self.model_path
            
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found. Please train the model first.")
            return None
            
        model = FallLSTM().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded from {model_path}")
        return model
    
    def get_predictions(self, model, X_data, batch_size=32):
        """Get model predictions and probabilities."""
        dataset = FallDataset(X_data, np.zeros(len(X_data)))  # Dummy labels
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_probs = []
        all_preds = []
        
        model.eval()
        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                probs = outputs.cpu().numpy().flatten()
                preds = (outputs > 0.5).float().cpu().numpy().flatten()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
        
        return np.array(all_preds), np.array(all_probs)
    
    def basic_validation(self, X_test, y_test, model_path=None):
        """Perform basic validation with detailed metrics."""
        print("\n" + "="*60)
        print("BASIC VALIDATION")
        print("="*60)
        
        model = self.load_model(model_path)
        if model is None:
            return None
        
        # Get predictions
        y_pred, y_probs = self.get_predictions(model, X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fall']))
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # ROC Curve
        self.plot_roc_curve(y_test, y_probs)
        
        # Precision-Recall Curve
        self.plot_precision_recall_curve(y_test, y_probs)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_probs': y_probs
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f'basic_validation_{timestamp}.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to: {results_file}")
        
        return results
    
    def cross_validation(self, X_data, y_data, cv_folds=5, model_class=FallLSTM):
        """Perform cross-validation."""
        print("\n" + "="*60)
        print(f"CROSS-VALIDATION ({cv_folds}-FOLD)")
        print("="*60)
        
        # Combine training and test data for CV
        X_combined = np.concatenate([X_data[0], X_data[1]], axis=0)
        y_combined = np.concatenate([y_data[0], y_data[1]], axis=0)
        
        print(f"Total samples for CV: {len(X_combined)}")
        print(f"Fall/Normal ratio: {np.sum(y_combined)}/{len(y_combined) - np.sum(y_combined)}")
        
        # Initialize cross-validator
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = []
        cv_precisions = []
        cv_recalls = []
        cv_f1s = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_combined, y_combined), 1):
            print(f"\nFold {fold}/{cv_folds}:")
            
            # Split data
            X_train_fold = X_combined[train_idx]
            X_val_fold = X_combined[val_idx]
            y_train_fold = y_combined[train_idx]
            y_val_fold = y_combined[val_idx]
            
            # Create datasets
            train_dataset = FallDataset(X_train_fold, y_train_fold)
            val_dataset = FallDataset(X_val_fold, y_val_fold)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            
            # Initialize and train model
            model = model_class().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCELoss()
            
            # Train for limited epochs (quick validation)
            model.train()
            for epoch in range(20):  # Quick training for CV
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate fold
            y_pred_fold, _ = self.get_predictions(model, X_val_fold)
            
            accuracy = accuracy_score(y_val_fold, y_pred_fold)
            precision = precision_score(y_val_fold, y_pred_fold, zero_division=0)
            recall = recall_score(y_val_fold, y_pred_fold, zero_division=0)
            f1 = f1_score(y_val_fold, y_pred_fold, zero_division=0)
            
            cv_scores.append(accuracy)
            cv_precisions.append(precision)
            cv_recalls.append(recall)
            cv_f1s.append(f1)
            
            print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Print CV results
        print(f"\nCross-Validation Results ({cv_folds}-fold):")
        print(f"Accuracy:  {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        print(f"Precision: {np.mean(cv_precisions):.4f} ± {np.std(cv_precisions):.4f}")
        print(f"Recall:    {np.mean(cv_recalls):.4f} ± {np.std(cv_recalls):.4f}")
        print(f"F1-Score:  {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")
        
        # Plot CV results
        self.plot_cv_results(cv_scores, cv_precisions, cv_recalls, cv_f1s)
        
        cv_results = {
            'cv_scores': cv_scores,
            'cv_precisions': cv_precisions,
            'cv_recalls': cv_recalls,
            'cv_f1s': cv_f1s,
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores)
        }
        
        return cv_results
    
    def temporal_validation(self, X_data, y_data, test_ratio=0.2):
        """Perform temporal validation (important for time-series data like fall detection)."""
        print("\n" + "="*60)
        print("TEMPORAL VALIDATION")
        print("="*60)
        
        # For temporal validation, we don't shuffle - keep temporal order
        X_combined = np.concatenate([X_data[0], X_data[1]], axis=0)
        y_combined = np.concatenate([y_data[0], y_data[1]], axis=0)
        
        # Split temporally (last 20% as test)
        split_idx = int(len(X_combined) * (1 - test_ratio))
        
        X_train_temp = X_combined[:split_idx]
        X_test_temp = X_combined[split_idx:]
        y_train_temp = y_combined[:split_idx]
        y_test_temp = y_combined[split_idx:]
        
        print(f"Temporal train: {len(X_train_temp)}, test: {len(X_test_temp)}")
        
        # Train model
        train_dataset = FallDataset(X_train_temp, y_train_temp)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model = FallLSTM().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        print("Training model for temporal validation...")
        model.train()
        for epoch in range(30):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        y_pred_temp, y_probs_temp = self.get_predictions(model, X_test_temp)
        
        accuracy = accuracy_score(y_test_temp, y_pred_temp)
        precision = precision_score(y_test_temp, y_pred_temp, zero_division=0)
        recall = recall_score(y_test_temp, y_pred_temp, zero_division=0)
        f1 = f1_score(y_test_temp, y_pred_temp, zero_division=0)
        
        print(f"Temporal Validation Results:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def learning_curve_analysis(self, X_train, y_train, X_val, y_val, epochs=100):
        """Analyze learning curves to detect overfitting."""
        print("\n" + "="*60)
        print("LEARNING CURVE ANALYSIS")
        print("="*60)
        
        # Create datasets
        train_dataset = FallDataset(X_train, y_train)
        val_dataset = FallDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Initialize model
        model = FallLSTM().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        print("Training model for learning curve analysis...")
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Plot learning curves
        self.plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies)
        
        # Detect overfitting
        overfitting_detected = self.detect_overfitting(train_losses, val_losses, train_accuracies, val_accuracies)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'overfitting_detected': overfitting_detected
        }
    
    def detect_overfitting(self, train_losses, val_losses, train_accs, val_accs, patience=10):
        """Detect overfitting based on learning curves."""
        print("\nOverfitting Analysis:")
        
        # Check if validation loss starts increasing while training loss decreases
        min_val_loss_epoch = np.argmin(val_losses)
        final_epochs = len(val_losses)
        
        overfitting_signals = []
        
        # Signal 1: Validation loss increasing in final epochs while training loss decreasing
        if min_val_loss_epoch < final_epochs - patience:
            overfitting_signals.append("Validation loss stopped improving early")
        
        # Signal 2: Large gap between train and validation accuracy
        final_train_acc = train_accs[-1]
        final_val_acc = val_accs[-1]
        acc_gap = final_train_acc - final_val_acc
        
        if acc_gap > 0.1:  # 10% gap threshold
            overfitting_signals.append(f"Large accuracy gap: {acc_gap:.3f}")
        
        # Signal 3: Training accuracy much higher than validation
        if final_train_acc > 0.95 and final_val_acc < 0.85:
            overfitting_signals.append("Training accuracy very high, validation low")
        
        if overfitting_signals:
            print("⚠️  Overfitting detected:")
            for signal in overfitting_signals:
                print(f"   • {signal}")
            print("\n💡 Recommendations:")
            print("   • Add more regularization (dropout, weight decay)")
            print("   • Reduce model complexity")
            print("   • Get more training data")
            print("   • Use early stopping")
            return True
        else:
            print("✅ No clear signs of overfitting detected")
            return False
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Fall'], 
                   yticklabels=['Normal', 'Fall'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if self.save_plots:
            plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_probs):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if self.save_plots:
            plt.savefig(os.path.join(self.results_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Area Under ROC Curve (AUC): {roc_auc:.4f}")
    
    def plot_precision_recall_curve(self, y_true, y_probs):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if self.save_plots:
            plt.savefig(os.path.join(self.results_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Area Under PR Curve: {pr_auc:.4f}")
    
    def plot_cv_results(self, scores, precisions, recalls, f1s):
        """Plot cross-validation results."""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [scores, precisions, recalls, f1s]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, (metric, vals) in enumerate(zip(metrics, values)):
            axes[i].boxplot(vals)
            axes[i].set_title(f'{metric} Distribution (CV)', fontsize=12)
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
            axes[i].text(0.02, 0.98, f'Mean: {np.mean(vals):.3f}\nStd: {np.std(vals):.3f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(os.path.join(self.results_dir, 'cv_results.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, train_losses, val_losses, train_accs, val_accs):
        """Plot learning curves."""
        epochs = range(1, len(train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Learning Curves - Loss', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Learning Curves - Accuracy', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(os.path.join(self.results_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def comprehensive_validation(self):
        """Run all validation methods."""
        print("🔍 COMPREHENSIVE MODEL VALIDATION")
        print("="*80)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        if X_train is None:
            return None
        
        validation_results = {}
        
        # 1. Basic Validation
        basic_results = self.basic_validation(X_test, y_test)
        validation_results['basic'] = basic_results
        
        # 2. Cross-Validation
        cv_results = self.cross_validation([X_train, X_test], [y_train, y_test])
        validation_results['cross_validation'] = cv_results
        
        # 3. Temporal Validation
        temporal_results = self.temporal_validation([X_train, X_test], [y_train, y_test])
        validation_results['temporal'] = temporal_results
        
        # 4. Learning Curve Analysis
        learning_results = self.learning_curve_analysis(X_train, X_test, y_train, y_test, epochs=50)
        validation_results['learning_curves'] = learning_results
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f'comprehensive_validation_{timestamp}.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(validation_results, f)
        
        print(f"\n📊 Comprehensive validation completed!")
        print(f"All results saved to: {self.results_dir}")
        print(f"Detailed results file: {results_file}")
        
        return validation_results

def quick_validate(model_path='best_fall_lstm.pth'):
    """Quick validation function for easy use."""
    validator = ModelValidator(model_path=model_path)
    X_train, X_test, y_train, y_test = validator.load_data()
    if X_train is not None:
        return validator.basic_validation(X_test, y_test)
    return None

def main():
    """Main validation function."""
    print("Fall Detection Model Validation")
    print("=" * 50)
    
    # Check for model files
    model_files = ['best_fall_lstm.pth', 'fall_lstm_final.pth']
    available_models = [f for f in model_files if os.path.exists(f)]
    
    if not available_models:
        print("❌ No trained models found! Please train the model first.")
        print("Available commands:")
        print("  python train.py")
        print("  python main.py --stage train")
        return
    
    print(f"✅ Found trained models: {available_models}")
    
    # Use best model if available, otherwise use final model
    model_path = 'best_fall_lstm.pth' if 'best_fall_lstm.pth' in available_models else available_models[0]
    print(f"Using model: {model_path}")
    
    # Create validator and run comprehensive validation
    validator = ModelValidator(model_path=model_path)
    results = validator.comprehensive_validation()
    
    if results:
        print("\n🎉 Validation completed successfully!")
        print("Check the 'validation_results' folder for detailed plots and metrics.")
    else:
        print("\n❌ Validation failed. Please check your data files.")

if __name__ == "__main__":
    main()
