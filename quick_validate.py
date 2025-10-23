#!/usr/bin/env python3
"""
Quick Validation Script for Fall Detection Model
==================================================

This script provides quick validation options for the fall detection model.

Usage:
    python quick_validate.py                 # Basic validation
    python quick_validate.py --full          # Comprehensive validation
    python quick_validate.py --model path    # Validate specific model
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Quick Fall Detection Model Validation')
    parser.add_argument('--model', type=str, default='best_fall_lstm.pth',
                       help='Path to model file (default: best_fall_lstm.pth)')
    parser.add_argument('--full', action='store_true',
                       help='Run comprehensive validation (includes CV, learning curves)')
    parser.add_argument('--plots', action='store_true', default=True,
                       help='Save validation plots (default: True)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model file '{args.model}' not found!")
        print("\nAvailable models:")
        model_files = ['best_fall_lstm.pth', 'fall_lstm_final.pth']
        available = [f for f in model_files if os.path.exists(f)]
        if available:
            for model in available:
                print(f"  ✓ {model}")
        else:
            print("  No trained models found. Please train a model first:")
            print("    python train.py")
            print("    python main.py --stage train")
        return
    
    try:
        from validate import ModelValidator, quick_validate
        
        print(f"🔍 Validating model: {args.model}")
        print("=" * 50)
        
        if args.full:
            print("Running comprehensive validation...")
            validator = ModelValidator(model_path=args.model, save_plots=args.plots)
            results = validator.comprehensive_validation()
            
            if results:
                print("\n🎉 Comprehensive validation completed!")
                print("Check 'validation_results' folder for detailed plots and metrics.")
            else:
                print("\n❌ Validation failed.")
        else:
            print("Running basic validation...")
            results = quick_validate(args.model)
            
            if results:
                print(f"\n📊 Quick Validation Results:")
                print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
                print(f"  Precision: {results['precision']:.4f}")
                print(f"  Recall:    {results['recall']:.4f}")
                print(f"  F1-Score:  {results['f1_score']:.4f}")
                print("\n💡 For more detailed analysis, use: python quick_validate.py --full")
            else:
                print("\n❌ Basic validation failed.")
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure all required packages are installed:")
        print("  pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Validation error: {e}")

if __name__ == "__main__":
    main()
