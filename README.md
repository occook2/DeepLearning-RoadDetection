# Deep Learning Portfolio: From MLPs to Transformers

A comprehensive deep learning portfolio showcasing progressive neural network architectures applied to computer vision and autonomous driving tasks. This project demonstrates the evolution from basic multi-layer perceptrons to advanced transformers and convolutional networks, all trained on the SuperTuxKart dataset.

## 🎯 Project Overview

This repository contains three interconnected projects that progressively explore modern deep learning architectures:

1. **MLP Image Classification** - Understanding depth and residual connections in fully-connected networks
2. **CNN Multi-Task Detection** - U-Net architecture for simultaneous segmentation and depth estimation
3. **Transformer Autonomous Planner** - Perceiver-style transformers and CNNs for trajectory prediction

### Learning Journey

The projects follow a natural progression in deep learning:
- **MLPs** → Understanding the basics: depth, activations, and residual connections
- **CNNs** → Spatial feature extraction and multi-task learning
- **Transformers** → Attention mechanisms and sequential decision-making

## 📋 Table of Contents

- [Installation](#installation)
- [Projects](#projects)
  - [1. MLP Image Classification](#1-mlp-image-classification)
  - [2. CNN Multi-Task Detection](#2-cnn-multi-task-detection)
  - [3. Transformer Autonomous Planner](#3-transformer-autonomous-planner)
- [Dataset](#dataset)
- [Results & Examples](#results--examples)
- [Project Structure](#project-structure)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA support for faster training
- (Optional) Apple Silicon Mac with MPS support

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/occook2/DeepLearning-RoadDetection.git
   cd DeepLearning-RoadDetection
   ```

2. **Install dependencies:**
   
   The project automatically detects and uses available hardware (CUDA > MPS > CPU).
   
   **For CPU-only:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **For NVIDIA GPU (CUDA 12.1):**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
   
   **For Apple Silicon (M1/M2/M3):**
   ```bash
   pip install torch torchvision
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
   ```

4. **Download datasets (Required):**
   
   Each project requires downloading its dataset before training. Run the download script for each project:
   
   ```bash
   # MLP Image Classification
   cd mlp-image-classification/src
   python download_data.py
   cd ../..
   
   # CNN Multi-Task Detection
   cd cnn-multitask-detection/src
   python download_data.py
   cd ../..
   
   # Transformer Autonomous Planner
   cd transformer-autonomous-planner/src
   python download_data.py
   cd ../..
   ```
   
   **Note:** The download scripts will automatically place datasets in the correct directories. The datasets are not included in the repository due to size constraints.

## 📚 Projects

### 1. MLP Image Classification

**Directory:** `mlp-image-classification/`

#### Description
Explores the impact of network depth and residual connections on image classification performance using fully-connected networks. Implements four architectures of increasing complexity:
- Linear Classifier (baseline)
- Simple MLP (1 hidden layer)
- Deep MLP (4+ layers)
- Deep Residual MLP (residual connections + LayerNorm)

#### Dataset
- **SuperTuxKart Classification Dataset** (64×64 RGB images)
- **6 classes:** background, kart, pickup, nitro, bomb, projectile
- **Splits:** train/validation/test

#### Models & Performance
| Model | Architecture | Parameters | Target Accuracy |
|-------|-------------|------------|-----------------|
| LinearClassifier | Single linear layer | ~74K | 70%+ |
| MLPClassifier | 1 hidden layer (64 units) | ~786K | 80%+ |
| MLPClassifierDeep | 4 hidden layers (16 units each) | ~200K | 80%+ |
| MLPClassifierDeepResidual | Residual blocks + LayerNorm | ~200K | 80%+ |

#### Quick Start

**1. Download the dataset:**
```bash
cd mlp-image-classification/src
python download_data.py
```

**2. Train a model:**
```bash
python train.py --model <model_name> --epochs 50 --lr 1e-3 --batch_size 256
```

**Available models:** `linear`, `mlp`, `mlp_deep`, `mlp_deep_residual`

#### Key Features
- Early stopping when validation accuracy > 82%
- TensorBoard logging for training visualization
- Timing analysis for data loading vs. model computation
- SGD optimizer with momentum (0.9)

#### Expected Outputs
- Trained model checkpoint: `<model_name>.th`
- TensorBoard logs in `logs/` directory
- Training/validation accuracy and loss curves
- Timing statistics

#### View Training Progress
```bash
tensorboard --logdir mlp-image-classification/logs
```

---

### 2. CNN Multi-Task Detection

**Directory:** `cnn-multitask-detection/`

#### Description
Implements convolutional neural networks for two tasks:
1. **CNN Classifier** - Basic CNN for image classification
2. **Multi-Task Detector** - U-Net style encoder-decoder for simultaneous road segmentation and depth estimation

#### Dataset
- **Classification:** Same as Project 1 (64×64 RGB, 6 classes)
- **Detection:** SuperTuxKart Drive Dataset (96×128 RGB)
  - **Segmentation:** 3 classes (background, left boundary, right boundary)
  - **Depth maps:** Normalized [0, 1]

#### Models & Performance

**CNN Classifier:**
- Architecture: Conv layers → Batch Normalization → Max Pooling → Adaptive Pooling
- Input normalization with learned mean/std
- Target accuracy: 80%+

**Multi-Task Detector:**
- U-Net architecture with encoder-decoder + skip connections
- Dual output heads:
  - Segmentation head (3-class classification)
  - Depth head (regression with sigmoid activation)
- Combined loss: `L_total = L_seg + 0.5 * L_depth`
- Target metrics:
  - Segmentation mIoU > 0.75
  - Depth MAE < 0.05
  - Boundary depth MAE < 0.05

#### Quick Start

**1. Download the datasets:**
```bash
cd cnn-multitask-detection/src
python download_data.py
```

**2. Train models:**

**Train CNN Classifier:**
```bash
python train_classification.py --epochs 50 --lr 1e-3 --batch_size 256
```

**Train Multi-Task Detector:**
```bash
python train_detection.py --epochs 100 --lr 1e-3 --batch_size 128
```

#### Key Features
- Batch normalization for stable training
- Skip connections for preserving spatial information
- Multi-task learning with weighted loss combination
- Confusion matrix for IoU calculation
- Adam optimizer

#### Expected Outputs
- Model checkpoints: `classifier.th`, `detector.th`
- TensorBoard logs with:
  - Classification accuracy
  - Segmentation mIoU
  - Depth MAE metrics
  - Combined training loss

---

### 3. Transformer Autonomous Planner

**Directory:** `transformer-autonomous-planner/`

#### Description
Implements three different architectures for autonomous driving trajectory prediction:
1. **MLP Planner** - Baseline fully-connected network
2. **Transformer Planner** - Perceiver-style architecture with cross-attention
3. **CNN Planner** - End-to-end vision-based planning with U-Net backbone

#### Dataset
- **SuperTuxKart Drive Dataset** with trajectory labels
- **Inputs:**
  - Track boundaries: (10, 2) lane boundary points
  - RGB images: (3, 96, 128) camera feed
- **Outputs:**
  - Waypoints: (3, 2) future trajectory positions
  - Validity mask: (3,) for valid waypoints

#### Models & Performance

| Model | Input | Architecture | Longitudinal Error | Lateral Error |
|-------|-------|--------------|-------------------|---------------|
| MLPPlanner | Track boundaries | 3-layer MLP | < 0.2 | < 0.6 |
| TransformerPlanner | Track boundaries | Perceiver decoder | < 0.2 | < 0.6 |
| CNNPlanner | RGB images | U-Net + global pooling | < 0.30 | < 0.45 |

#### Architecture Highlights

**MLP Planner:**
- Input: Flattened track boundaries (40 features)
- Hidden layers: 64 → 16 units
- Output: 6 values reshaped to (3, 2) waypoints

**Transformer Planner:**
- Learned query embeddings for waypoints
- Cross-attention over lane boundary features
- TransformerDecoder with positional encoding
- Output projection to waypoint coordinates

**CNN Planner:**
- Encoder-decoder for feature extraction
- Global average pooling for scene understanding
- Direct regression to waypoint coordinates
- End-to-end learning from pixels

#### Quick Start

**1. Download the dataset:**
```bash
cd transformer-autonomous-planner/src
python download_data.py
```

**2. Train models:**

**Train MLP Planner:**
```bash
python train_planner.py --model mlp --epochs 50 --lr 1e-3 --batch_size 256
```

**Train Transformer Planner:**
```bash
python train_transformer.py --epochs 50 --lr 5e-4 --batch_size 256
```

**Train CNN Planner:**
```bash
python train_cnn_planner.py --epochs 100 --lr 1e-3 --batch_size 128
```

#### Key Features
- Waypoint masking for variable-length trajectories
- L1 loss for trajectory regression
- Input normalization (learned statistics)
- Adam optimizer with architecture-specific learning rates
- Video visualization of driving behavior

#### Expected Outputs
- Model checkpoints: `mlp_planner.th`, `transformer_planner.th`, `cnn_planner.th`
- TensorBoard logs with longitudinal/lateral error metrics
- (Optional) Driving visualization videos in `videos/` directory

#### Visualization
The project includes tools for visualizing autonomous driving behavior:
```bash
# Generate driving videos (requires trained model)
python utils.py --visualize --model cnn_planner.th
```

---

## 📊 Dataset

All projects use the **SuperTuxKart** dataset, a synthetic racing game dataset with diverse environments:

- **Tracks:** Cornfield Crossing, Hacienda, Lighthouse, Snow Mountain, Zen Garden
- **Weather conditions:** Clear, rainy, foggy
- **Time of day:** Day, sunset, night
- **Camera poses:** Multiple viewpoints

The dataset is automatically downloaded when running training scripts.

## 🎨 Results & Examples

Sample training outputs, loss curves, and visualizations are provided in the `examples/` directory:

```
examples/
├── mlp-image-classification/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── model_comparison.png
├── cnn-multitask-detection/
│   ├── segmentation_examples.png
│   ├── depth_maps.png
│   └── multitask_loss_curves.png
└── transformer-autonomous-planner/
    ├── trajectory_predictions.png
    ├── driving_video_sample.mp4
    └── planner_comparison.png
```

**Note:** You can generate your own results by running the training scripts. Example outputs will be saved automatically during training.

## 📁 Project Structure

```
DeepLearning-RoadDetection/
├── README.md                           # This file
├── requirements.txt                     # Unified dependencies
├── examples/                            # Sample outputs and visualizations
│   ├── mlp-image-classification/
│   ├── cnn-multitask-detection/
│   └── transformer-autonomous-planner/
│
├── mlp-image-classification/            # Project 1: MLPs
│   ├── README.md                        # Original homework description
│   ├── requirements.txt                 # Project-specific requirements
│   ├── logs/                            # TensorBoard logs
│   └── homework/
│       ├── models.py                    # Model definitions
│       ├── train.py                     # Training script
│       ├── logger.py                    # TensorBoard logging
│       ├── utils.py                     # Utilities
│       └── *.th                         # Trained model checkpoints
│
├── cnn-multitask-detection/             # Project 2: CNNs
│   ├── README.md
│   ├── requirements.txt
│   ├── logs/
│   └── homework/
│       ├── models.py                    # CNN and Detector models
│       ├── train_classification.py      # CNN training
│       ├── train_detection.py           # Multi-task training
│       ├── metrics.py                   # Evaluation metrics
│       ├── utils.py
│       ├── datasets/                    # Dataset loaders
│       └── *.th                         # Model checkpoints
│
└── transformer-autonomous-planner/      # Project 3: Transformers
    ├── README.md
    ├── requirements.txt
    ├── logs/
    ├── videos/                          # Driving visualizations
    └── homework/
        ├── models.py                    # MLP, Transformer, CNN planners
        ├── train_planner.py             # MLP training
        ├── train_transformer.py         # Transformer training
        ├── train_cnn_planner.py         # CNN training
        ├── metrics.py                   # Trajectory metrics
        ├── utils.py                     # Visualization tools
        ├── datasets/                    # Dataset loaders
        └── *.th                         # Model checkpoints
```

## 🛠️ Technical Details

### Device Compatibility

All training scripts automatically detect and use the best available hardware:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")      # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")        # Apple Silicon
else:
    device = torch.device("cpu")        # CPU fallback
```

### Platform Support

- ✅ **Windows** - Tested on Windows 10/11 with CPU and CUDA
- ✅ **macOS** - Compatible with Intel and Apple Silicon Macs
- ✅ **Linux** - Ubuntu, Debian, and other distributions

### Performance Tips

- **CUDA GPU:** Training is 10-20x faster on NVIDIA GPUs
- **Apple Silicon:** MPS provides 3-5x speedup over CPU on M1/M2/M3
- **CPU:** All models train successfully on CPU, but expect longer training times
- **Batch Size:** Reduce if you encounter out-of-memory errors

## 📝 Common Issues & Solutions

### Issue: Out of Memory
**Solution:** Reduce batch size in training command:
```bash
python train.py --batch_size 64  # Instead of 256
```

### Issue: Slow Training on Mac
**Solution:** Ensure MPS is being used:
```python
import torch
print(torch.backends.mps.is_available())  # Should print True
```

### Issue: Module Not Found
**Solution:** Ensure you're in the correct directory and have activated your environment:
```bash
cd <project-directory>/homework
python train.py
```

## 🎓 Learning Outcomes

This portfolio demonstrates proficiency in:

- ✅ **Neural Network Architectures:** MLPs, CNNs, Transformers, U-Nets
- ✅ **Computer Vision:** Image classification, semantic segmentation, depth estimation
- ✅ **Multi-Task Learning:** Joint optimization of related tasks
- ✅ **Attention Mechanisms:** Transformer decoders and cross-attention
- ✅ **Training Best Practices:** Normalization, residual connections, early stopping
- ✅ **Experiment Tracking:** TensorBoard integration
- ✅ **Code Quality:** Modular design, device compatibility, reproducibility

## 📧 Contact

**Author:** [Your Name]
**GitHub:** [@occook2](https://github.com/occook2)
**LinkedIn:** [Your LinkedIn Profile]
**Email:** your.email@example.com

---

## 🙏 Acknowledgments

- Dataset: SuperTuxKart open-source racing game
- Course: UT Austin's MSAI Program's Deep Learning Course (Master's Program)
- Framework: PyTorch

## 📄 License

This project is available for educational and portfolio purposes. Please credit if you use any part of this code.

---

**⭐ If you find this project helpful, please consider starring the repository!**
