# MLP Image Classification - Example Outputs

This directory contains sample outputs and visualizations from the MLP image classification project.

## Expected Contents

### Training Curves
- `training_curves.png` - Training and validation accuracy/loss over epochs
- `model_comparison.png` - Comparison of different MLP architectures (Linear, MLP, Deep, Residual)

### Model Performance
- `confusion_matrix.png` - Confusion matrix showing per-class performance
- `sample_predictions.png` - Grid of sample images with predicted vs. actual labels

### Architecture Analysis
- `residual_vs_deep.png` - Comparison of deep MLP with and without residual connections
- `depth_analysis.png` - Impact of network depth on performance

## How to Generate These Outputs

1. **Train all models:**
   ```bash
   cd ../mlp-image-classification/homework
   python train.py --model linear --epochs 50
   python train.py --model mlp --epochs 50
   python train.py --model mlp_deep --epochs 50
   python train.py --model mlp_deep_residual --epochs 50
   ```

2. **View TensorBoard logs:**
   ```bash
   tensorboard --logdir ../mlp-image-classification/logs
   ```
   Take screenshots of the training curves and save them here.

3. **Generate visualizations:**
   - Use matplotlib to create confusion matrices
   - Plot sample predictions
   - Compare model performances

## Notes

These are example outputs to showcase the project. Recruiters can see model performance without running the full training pipeline.
