"""
Fall Detection Pipeline
======================

This pipeline consists of three main stages:
1. Load Data: Extract features from video files using pre-trained ResNet
2. Process Data: Create sequences and prepare datasets for training
3. Train: Train the LSTM model for fall detection

Usage:
    python main.py --stage all          # Run all stages
    python main.py --stage load         # Only load data
    python main.py --stage process      # Only process data
    python main.py --stage train        # Only train model
    python main.py --evaluate           # Evaluate saved model
"""

import argparse
import os
import sys

def run_load_data():
    """Run the data loading stage."""
    print("=" * 50)
    print("STAGE 1: LOADING DATA")
    print("=" * 50)
    
    try:
        from load_data import load_raw_data
        import numpy as np
        
        print("Loading raw data...")
        features, labels = load_raw_data()
        print(f"Loaded {len(features)} video files")
        
        # Save raw data as object arrays to handle different shapes
        np.save('raw_features.npy', np.array(features, dtype=object), allow_pickle=True)
        np.save('raw_labels.npy', np.array(labels, dtype=object), allow_pickle=True)
        print("Raw data saved to raw_features.npy and raw_labels.npy")
        return True
        
    except Exception as e:
        print(f"Error in data loading stage: {e}")
        return False

def run_process_data():
    """Run the data processing stage."""
    print("=" * 50)
    print("STAGE 2: PROCESSING DATA")
    print("=" * 50)
    
    try:
        from process_data import process_data
        import numpy as np
        
        X_train, X_test, y_train, y_test, train_dataset, test_dataset = process_data()
        
        if X_train is not None:
            # Save processed data
            np.save('X_train.npy', X_train)
            np.save('X_test.npy', X_test)
            np.save('y_train.npy', y_train)
            np.save('y_test.npy', y_test)
            print("Processed data saved to X_train.npy, X_test.npy, y_train.npy, y_test.npy")
            return True
        else:
            print("Data processing failed.")
            return False
            
    except Exception as e:
        print(f"Error in data processing stage: {e}")
        return False

def run_training():
    """Run the training stage."""
    print("=" * 50)
    print("STAGE 3: TRAINING MODEL")
    print("=" * 50)
    
    try:
        from train import train_model
        
        model = train_model(epochs=100, batch_size=32, learning_rate=0.001)
        
        if model is not None:
            print("\nTraining completed successfully!")
            print("Models saved:")
            print("- best_fall_lstm.pth (best performing model during training)")
            print("- fall_lstm_final.pth (final model after all epochs)")
            return True
        else:
            print("Training failed. Please check the data files.")
            return False
            
    except Exception as e:
        print(f"Error in training stage: {e}")
        return False

def run_evaluation():
    """Run model evaluation."""
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    try:
        from train import load_and_evaluate
        
        if os.path.exists('best_fall_lstm.pth'):
            print("Evaluating best model...")
            load_and_evaluate('best_fall_lstm.pth')
        elif os.path.exists('fall_lstm_final.pth'):
            print("Evaluating final model...")
            load_and_evaluate('fall_lstm_final.pth')
        else:
            print("No trained model found. Please run training first.")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return False

def run_validation():
    """Run comprehensive model validation."""
    print("=" * 50)
    print("COMPREHENSIVE MODEL VALIDATION")
    print("=" * 50)
    
    try:
        from validate import ModelValidator
        
        # Check for model files
        model_files = ['best_fall_lstm.pth', 'fall_lstm_final.pth']
        available_models = [f for f in model_files if os.path.exists(f)]
        
        if not available_models:
            print("❌ No trained models found! Please train the model first.")
            return False
        
        # Use best model if available
        model_path = 'best_fall_lstm.pth' if 'best_fall_lstm.pth' in available_models else available_models[0]
        print(f"Using model: {model_path}")
        
        # Create validator and run comprehensive validation
        validator = ModelValidator(model_path=model_path)
        results = validator.comprehensive_validation()
        
        if results:
            print("✅ Comprehensive validation completed successfully!")
            return True
        else:
            print("❌ Validation failed.")
            return False
            
    except Exception as e:
        print(f"Error in validation: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Fall Detection Pipeline')
    parser.add_argument('--stage', choices=['all', 'load', 'process', 'train'], 
                       default='all', help='Which stage to run')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Evaluate trained model (basic metrics)')
    parser.add_argument('--validate', action='store_true',
                       help='Run comprehensive model validation')
    
    args = parser.parse_args()
    
    if args.evaluate:
        run_evaluation()
        return
    
    if args.validate:
        run_validation()
        return
    
    success = True
    
    if args.stage in ['all', 'load']:
        success = run_load_data() and success
        if not success and args.stage == 'all':
            print("Data loading failed. Stopping pipeline.")
            return
    
    if args.stage in ['all', 'process']:
        success = run_process_data() and success
        if not success and args.stage == 'all':
            print("Data processing failed. Stopping pipeline.")
            return
    
    if args.stage in ['all', 'train']:
        success = run_training() and success
    
    if success:
        print("\n" + "=" * 50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("PIPELINE FAILED!")
        print("=" * 50)

if __name__ == "__main__":
    main()
