# Transformer Autonomous Planner - Example Outputs

This directory contains sample outputs and visualizations from the transformer autonomous driving planner project.

## Expected Contents

### Trajectory Predictions
- `trajectory_predictions.png` - Visualization of predicted waypoints overlaid on track boundaries
- `planner_comparison.png` - Side-by-side comparison of MLP, Transformer, and CNN planner outputs

### Performance Metrics
- `longitudinal_error.png` - Longitudinal error curves for all three planners
- `lateral_error.png` - Lateral error curves for all three planners
- `error_comparison.png` - Bar chart comparing final metrics across planners

### Driving Visualizations
- `driving_video_sample.mp4` - Short video clip showing autonomous driving behavior
- `trajectory_overlay.png` - RGB images with predicted waypoints and actual trajectory
- `attention_visualization.png` - (Transformer only) Attention weights over lane boundary features

### Architecture Comparison
- `mlp_predictions.png` - MLP planner behavior on various tracks
- `transformer_predictions.png` - Transformer planner with learned attention
- `cnn_predictions.png` - End-to-end vision-based CNN planner

### Failure Analysis
- `challenging_scenarios.png` - Sharp turns, intersections, and edge cases
- `track_comparison.png` - Performance across different tracks (cornfield, hacienda, lighthouse, snowmountain)

## How to Generate These Outputs

1. **Train all three planners:**
   ```bash
   cd ../transformer-autonomous-planner/src

   # MLP Planner
   python train_planner.py --model mlp --num_epoch 50

   # Transformer Planner
   python train_transformer.py --num_epoch 50

   # CNN Planner
   python train_cnn_planner.py --num_epoch 100
   ```

2. **View TensorBoard:**
   ```bash
   tensorboard --logdir ../transformer-autonomous-planner/logs
   ```

3. **Generate driving videos:**
   ```bash
   cd ../transformer-autonomous-planner
   python -m src.supertux_utils.evaluate --model transformer_planner --track lighthouse --max-steps 200
   # Output saved to videos/transformer_planner_lighthouse.mp4

   # Other available models: mlp_planner, cnn_planner
   # Other available tracks: cornfield_crossing, hacienda, snowmountain, zengarden
   ```

4. **Create comparison plots:**
   - Plot waypoints on track boundaries
   - Overlay predicted vs. actual trajectories
   - Compare error metrics across models

## Notes

This project demonstrates the evolution from simple MLPs to attention-based transformers and end-to-end vision systems. The Transformer planner learns to attend to relevant lane boundary features, while the CNN planner performs end-to-end learning from raw pixels to control outputs.

### Key Insights
- **MLP:** Simple and fast, but limited to track boundary features
- **Transformer:** Attention mechanism helps focus on relevant parts of the track
- **CNN:** End-to-end learning from images, more robust to visual variations
