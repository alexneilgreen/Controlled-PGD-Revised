# Controlled PGD (CPGD) Project

Implementation of Controlled Projected Gradient Descent for targeted adversarial attacks on image classification models.

## Project Overview

This project implements both standard PGD (untargeted) and Controlled PGD (targeted) attacks on deep learning models. CPGD allows you to specify exactly which class each input should be misclassified as, enabling more controlled adversarial testing.

## Directory Structure

```
project/
├── Architecture/
│   ├── ResNet.py         # ResNet-18 implementation
│   └── ViT.py            # Vision Transformer implementation
├── Attacks/
│   ├── Classes.py        # Attack wrapper classes
│   ├── PGD.py            # Standard PGD attack
│   └── CPGD.py           # Controlled PGD attack
├── Data/                 # Dataset storage
├── Models/               # Saved trained models
├── Results/
│   └── reporter.py       # Metrics reporting
├── Data_Loader.py        # Dataset loading utilities
├── Main.py               # Main execution script
└── Setup_Directories.py  # Setup script
```

## Setup

1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

2. **Setup Directory Structure**

```bash
python Setup_Directories.py
```

## Usage

### Training Models

Train all models on all datasets:

```bash
python Main.py --mode train --model all --dataset all --epochs 50 --lr 0.001 --batch_size 128
```

Train specific model on specific dataset:

```bash
python Main.py --mode train --model resnet --dataset cifar10 --epochs 50
```

Options for `--model`: `resnet`, `vit`, `all`
Options for `--dataset`: `mnist`, `cifar10`, `cifar100`, `stl10`, `all`

Retrain existing models:

```bash
python Main.py --mode train --model all --dataset all --retrain
```

### Running Attacks

Run attacks (interactive mode):

```bash
python Main.py --mode attack --batch_size 128 --iterations 100 --attack_lr 0.01
```

The script will:

1. Show available trained models
2. Let you select a model
3. Let you choose attack type (PGD or CPGD)
4. For CPGD: prompt for class mapping
5. Display attack results with metrics

### Attack Parameters

- `--iterations`: Number of attack iterations (default: 100)
- `--tolerance`: Convergence tolerance (default: 0.000001)
- `--attack_lr`: Attack step size (default: 0.01)
- `--batch_size`: Batch size for evaluation (default: 128)

## Class Mapping for CPGD

When using CPGD, you'll be prompted to enter a class mapping. For example, for CIFAR-10:

```
Please input Matrix Mapping Values
Class 0 -> 5    # Map all class 0 samples to class 5
Class 1 -> 5    # Map all class 1 samples to class 5
Class 2 -> 3    # Map all class 2 samples to class 3
...
```

Common mapping patterns:

- **All to one**: Map all classes to a single target class
- **Sequential shift**: Map class i to class (i+1) % num_classes
- **Conditional**: Map dangerous classes to safe classes (e.g., stop sign → speed limit)

## Metrics

The system reports three key metrics:

### 1. Global Attack Success Rate (GASR)

Overall percentage of samples successfully misclassified.

### 2. Individual Attack Success Rate (IASR)

Per-class attack success rates showing which classes are more vulnerable.

### 3. Accuracy

Classification accuracy after attack (Accuracy = 1 - GASR).

### 4. Targeted Success Rate (CPGD only)

Percentage of samples misclassified to the intended target class.

## Example Workflow

```bash
# 1. Train ResNet on CIFAR-10
python Main.py --mode train --model resnet --dataset cifar10 --epochs 50

# 2. Run untargeted PGD attack
python Main.py --mode attack
# Select the trained model
# Choose option 1 (PGD)

# 3. Run targeted CPGD attack
python Main.py --mode attack
# Select the trained model
# Choose option 2 (CPGD)
# Enter class mappings when prompted
```

## Datasets

Supported datasets:

- **MNIST**: 10 classes, grayscale digits
- **CIFAR-10**: 10 classes, color images
- **CIFAR-100**: 100 classes, color images
- **STL-10**: 10 classes, high-resolution color images

All images are resized to 96×96 for consistent processing.

## Models

### ResNet-18

- Convolutional neural network with residual connections
- 4 layers with skip connections
- Efficient for image classification

### Vision Transformer (ViT)

- Transformer-based architecture
- Patches images into 8×8 patches
- 6 transformer blocks with 6 attention heads
- Embed dimension: 384

## Technical Details

### PGD Attack

- Untargeted attack maximizing loss
- Iterative gradient ascent
- L-infinity norm constraint (ε = 0.3)
- Step size: α = 0.01

### CPGD Attack

- Targeted attack minimizing loss for target class
- User-defined class mappings
- Same constraints as PGD
- Gradient descent toward target

## Troubleshooting

**Issue**: "No trained models found"
**Solution**: Train models first using `--mode train`

**Issue**: CUDA out of memory
**Solution**: Reduce `--batch_size` parameter

**Issue**: Import errors
**Solution**: Run `Setup_Directories.py` and ensure all `__init__.py` files exist

**Issue**: Slow training
**Solution**: Reduce `--epochs` or use a GPU-enabled system

## Future Enhancements

- Multi-victim, multi-target mappings
- Additional model architectures (LLM, LVLM)
- Adaptive attack parameters
- Defense mechanisms
- Ablation studies

## Citation

If you use this code for research, please cite:

```
Controlled PGD: Targeted Adversarial Attacks for Image Classification
Alexander Green and Ernest Edwin Wheaton III
University of Central Florida, 2025
```

## License

This project is for educational and research purposes.
