# SnoutNet: Pet Nose Keypoint Localization

A PyTorch deep learning project for localizing pet nose coordinates $(u, v)$ in RGB images from the **Oxford-IIIT Pet Dataset**. The codebase supports training and evaluating a custom convolutional regressor (**SnoutNet**), transfer learning with pretrained **AlexNet** and **VGG16** backbones, geometric/photometric data augmentation, and multi-model ensembling with simplex-optimized weights.

---

## Project Structure

```text
.
├── oxfordDataset.py        # Custom PyTorch Dataset with aspect-preserving letterboxing & label-aware augmentations
├── model.py                # Custom SnoutNet CNN architecture (3 Conv blocks + 3 FC layers -> (u, v))
├── PretrainedModel.py      # Transfer learning builders for AlexNet and VGG16 regression heads
├── Train.py                # CLI training pipeline with MSE loss, Adam, ReduceLROnPlateau, and loss plotting
├── Test.py                 # CLI evaluation, Euclidean pixel error metrics, ensemble weighting, and visualizations
├── SnoutNet.txt            # Reference CLI commands for custom SnoutNet training & testing
├── SnoutNet-A.txt          # Reference CLI commands for AlexNet backbone training & testing
├── SnoutNet-V.txt          # Reference CLI commands for VGG16 backbone training & testing
└── SnoutNet-Ensemble.txt   # Reference CLI commands for multi-model ensemble evaluation & visualization
```

---

## Model Architectures

1. **SnoutNet (`model.py`)**
   * **Feature Extractor:** Three convolutional blocks (`3 -> 64 -> 128 -> 256` channels, $3\times3$ kernels, padding $1$) with ReLU activations and $3\times3$ MaxPooling (`stride=4`), reducing a $227\times227\times3$ input to a $4\times4\times256$ feature map.
   * **Regressor Head:** Flattens the $4096$-dimensional feature vector through two $1024$-unit fully connected layers with ReLU into a 2D output predicting pixel coordinates $(u, v)$.

2. **Pretrained Backbones (`PretrainedModel.py`)**
   * Supports **AlexNet** (`SnoutNet-A`) and **VGG16** (`SnoutNet-V`) initialized with ImageNet weights (`IMAGENET1K_V1`).
   * Replaces the final classifier layer (`in_features = 4096`) with a 2-output linear regression head and supports freezing convolutional/feature layers during training.

3. **Multi-Model Ensemble (`Test.py`)**
   * Combines predictions across any subset of trained models (`snoutnet`, `alexnet`, `vgg16`).
   * Supports **equal weighting**, **manual weights** (`--ens-weights 1,1,2`), or **automatic convex weight optimization** (`--ens-weights auto`) learned via softmax-parameterized Adam optimization on a held-out validation split.

---

## Dataset & Augmentations (`oxfordDataset.py`)

Expected dataset layout under `data/oxford-iiit-pet-noses/`:
```text
data/oxford-iiit-pet-noses/
├── images-original/
│   └── images/             # Raw pet images
├── train_noses.txt         # Annotations formatted as: filename,"(u, v)"
└── test_noses.txt          # Test split annotations formatted as: filename,"(u, v)"
```

* **Coordinate-Consistent Resizing:** Supports both direct square resizing (`keep_aspect=False`) and aspect-preserving letterboxing with zero-padding (`keep_aspect=True`), automatically scaling and shifting ground-truth $(u, v)$ coordinates.
* **Label-Aware Augmentations (`-a 1`):**
  * **Horizontal Flip (`aug_hflip_p=0.5`):** Mirrors the image and updates $u \leftarrow (S - 1 - u)$.
  * **90° Counter-Clockwise Rotations (`aug_rotate=True`):** Randomly rotates by $0^\circ, 90^\circ, 180^\circ,$ or $270^\circ$ and transforms $(u, v)$ accordingly.
  * **Color Jitter:** Random brightness, contrast, and saturation adjustments ($0.2$).

---

## Requirements & Installation

* **Python:** 3.10+
* **Dependencies:**
  ```bash
  pip install torch torchvision matplotlib pillow
  ```
* **Hardware Acceleration:** Automatically selects Apple Silicon (`mps`), NVIDIA (`cuda`), or `cpu`.

---

## Usage

### 1. Training Models (`Train.py`)

**Custom SnoutNet (Baseline & Augmented):**
```bash
# Non-augmented training
python Train.py -e 20 -b 32 -s SnoutNet.pth -p SnoutNet_plot.png

# Augmented training (horizontal flip + 90° rotations + color jitter)
python Train.py -e 20 -b 32 -s aug_SnoutNet.pth -p aug_SnoutNet_plot.png -a 1
```

**Pretrained AlexNet (`SnoutNet-A`):**
```bash
# Non-augmented
python Train.py -e 20 -b 32 -s SnoutNet-A.pth -p SnoutNet-A_plot.png -t 1 --backbone alexnet

# Augmented
python Train.py -e 20 -b 32 -s aug_SnoutNet-A.pth -p aug_SnoutNet-A_plot.png -a 1 -t 1 --backbone alexnet
```

**Pretrained VGG16 (`SnoutNet-V`):**
```bash
# Non-augmented
python Train.py -e 20 -b 32 -s SnoutNet-V.pth -p SnoutNet-V_plot.png -t 1 --backbone vgg16

# Augmented
python Train.py -e 20 -b 32 -s aug_SnoutNet-V.pth -p aug_SnoutNet-V_plot.png -a 1 -t 1 --backbone vgg16
```

### 2. Single-Model Evaluation & Visualization (`Test.py`)

Evaluates checkpoints on the test split and reports **mean, standard deviation, min, and max Euclidean pixel error** ($\|\hat{y} - y\|_2$):

```bash
# Evaluate custom SnoutNet
python Test.py -c aug_SnoutNet.pth

# Evaluate pretrained AlexNet or VGG16
python Test.py -c aug_SnoutNet-A.pth -t 1 --backbone alexnet
python Test.py -c aug_SnoutNet-V.pth -t 1 --backbone vgg16

# Save prediction visualizations (purple dot = predicted, green dot = ground truth)
python Test.py -c aug_SnoutNet.pth --viz --viz-count 6 --viz-dir viz
```

### 3. Ensemble Evaluation (`Test.py`)

Combine multiple checkpoints and optionally optimize model weights automatically:

```bash
# Equal-weight ensemble evaluation (augmented models)
python Test.py --ensemble "snoutnet:aug_SnoutNet.pth,alexnet:aug_SnoutNet-A.pth,vgg16:aug_SnoutNet-V.pth"

# Learned simplex weights + prediction visualizations
python Test.py --ensemble "snoutnet:aug_SnoutNet.pth,alexnet:aug_SnoutNet-A.pth,vgg16:aug_SnoutNet-V.pth" \
  --ens-weights auto --viz --viz-count 5 --viz-dir ./viz_out
```

---

## Key CLI Arguments

| Script | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `Train.py` | `-e`, `--epochs` | `10` | Number of training epochs |
| `Train.py` | `-b`, `--batch_size` | `32` | Mini-batch size |
| `Train.py` | `-a`, `--augment` | `0` | Enable (`1`) or disable (`0`) geometric/photometric augmentations |
| `Train.py` | `-t`, `--toggle_pretrain` | `0` | Use custom `SnoutNet` (`0`) or pretrained backbone (`1`) |
| `Train.py` | `--backbone` | `alexnet` | Pretrained architecture (`alexnet` or `vgg16`) |
| `Train.py` | `--lr` | `1e-2` | Initial Adam learning rate (paired with `ReduceLROnPlateau`) |
| `Test.py` | `-c`, `--ckpt` | `aug_SnoutNet.pth` | Path to model `.pth` checkpoint |
| `Test.py` | `--ensemble` | `""` | Comma-separated `backbone:ckpt_path` pairs |
| `Test.py` | `--ens-weights` | `""` | Comma-separated weights (e.g., `1,1,2`) or `auto` for validation optimization |
| `Test.py` | `--viz` | `False` | Export visual overlays showing predicted $(u, v)$ vs. ground truth |