# CNN Multi-Task Detection - Example Outputs

This directory contains sample outputs and visualizations from the CNN multi-task detection project.

## Expected Contents

### Classification Results
- `classification_accuracy.png` - Training curves for CNN classifier
- `classification_samples.png` - Sample predictions on test images

### Segmentation Results
- `segmentation_examples.png` - Side-by-side comparison of input images, ground truth, and predicted segmentation masks
- `segmentation_iou.png` - Per-class IoU scores over training

### Depth Estimation Results
- `depth_maps.png` - Comparison of ground truth and predicted depth maps
- `depth_mae_curves.png` - Mean Absolute Error over training epochs

### Multi-Task Learning
- `multitask_loss_curves.png` - Combined loss and individual task losses over time
- `task_tradeoff.png` - Analysis of segmentation vs. depth performance tradeoff

### Qualitative Results
- `road_boundary_detection.png` - Examples showing left/right boundary detection
- `failure_cases.png` - Analysis of challenging scenarios

## How to Generate These Outputs

1. **Train CNN Classifier:**
   ```bash
   cd ../cnn-multitask-detection/homework
   python train_classification.py --epochs 50
   ```

2. **Train Multi-Task Detector:**
   ```bash
   python train_detection.py --epochs 100
   ```

3. **View TensorBoard:**
   ```bash
   tensorboard --logdir ../cnn-multitask-detection/logs
   ```

4. **Generate visualizations:**
   - Plot segmentation masks with color overlays
   - Visualize depth maps as heatmaps
   - Create comparison grids

## Notes

U-Net architecture with skip connections enables effective multi-task learning. The dual-head design allows simultaneous segmentation and depth estimation with shared feature extraction.
