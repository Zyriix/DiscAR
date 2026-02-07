# DiscAR: Discrete Autoregressive Image Generation

A clean and modular framework for training discrete autoregressive image generation models with modern techniques.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)

---

## Features

### 1. Discrete Tokenizers
- **Multiple Quantization Methods**: VQ-VAE, LFQ , IBQ
- **1D Token Architecture**: Flattened spatial tokens for sequence modeling
- **Tail Dropout**: Progressive context masking during training
- **Noise Query**: Diffusion-style denoising for decoder training

### 2. Modern Transformer Architecture
- **RMSNorm**: Efficient normalization
- **GEGLU**: Gated activation function
- **AdaLN**: Adaptive layer normalization with conditioning
- **RoPE**: Rotary position embeddings (optional)
- **Flexible Context**: Support for concat and none modes

### 3. Training Features
- **Two-Stage Training**: Separate tokenizer (AE) and prior (AR) training
- **Multiple Loss Functions**: L1, L2, LPIPS, optional GAN
- **Mixed Precision**: BF16/FP16 support
- **Multi-GPU**: Distributed training with Lightning Fabric
- **WandB Integration**: Experiment tracking and visualization
- **Torch Compile**: JIT compilation for speedup

> **Note**: This framework has been tested on **CIFAR-10** (32x32). Further verification on larger datasets like **ImageNet** is planned for future updates.

---

## Quick Start

### Installation

```bash
git clone https://github.com/Zyriix/DiscAR.git
cd DiscAR

# Create environment
conda env create -f environment.yml
conda activate DiscAR
```

### Data Preparation

Organize dataset:
```
./data/cifar10/
├── train/
└── test/
```

### Training

**Stage 1: Train Tokenizer**
```bash
python train.py --config=configs/CIFAR10_VQ_ae.yaml \
    data_dir=./data/cifar10
```

**Stage 2: Train AR Prior**
```bash
python train.py --config=configs/CIFAR10_VQ_ar.yaml \
    ae_ckpt_path=./checkpoints/tokenizer.ckpt
```

### Evaluation

```bash
# Evaluate reconstruction
python train.py --config=configs/CIFAR10_VQ_eval_ae.yaml \
    ae_ckpt_path=./checkpoints/tokenizer.ckpt

# Evaluate generation
python train.py --config=configs/CIFAR10_VQ_eval_ar.yaml \
    ae_ckpt_path=./checkpoints/tokenizer.ckpt \
    ar_ckpt_path=./checkpoints/ar_model.ckpt
```

## Project Structure

```
DiscAR/
├── configs/           # YAML configuration files
├── models.py          # Model architectures
├── train.py           # Training script
├── dataset.py         # Data loading
├── gan_loss.py        # GAN loss implementation
├── lpips.py           # Perceptual loss
└── calc_fid.py        # FID evaluation
```

---

## References

This project is inspired by and builds upon the following works:

- **[EDM](https://github.com/NVlabs/edm)** - NVIDIA (CC-BY-NC-SA-4.0)
- **[ADM](https://github.com/openai/guided-diffusion)** - OpenAI (MIT)
- **[VAR](https://github.com/FoundationVision/VAR)** - FoundationVision (MIT)
- **[FlowMo](https://github.com/kylesargent/FlowMo)** - Kyle Sargent (MIT)
- **[SEED-Voken](https://github.com/TencentARC/SEED-Voken)** - TencentARC (Apache 2.0)
- **[Taming Transformers](https://github.com/CompVis/taming-transformers)** - CompVis (MIT)
- **[ImageFolder](https://github.com/lxa9867/ImageFolder)** - lxa9867 (MIT)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
Copyright (c) 2026 Bowen Zheng
The Chinese University of Hong Kong, Shenzhen

Licensed under the MIT License.
```

---

## Contact
- Issues: [GitHub Issues](https://github.com/zyriix/DiscAR/issues)
- Author: Bowen Zheng
- Institution: The Chinese University of Hong Kong, Shenzhen
- Email: zyriix213 at gmail.com
