#!/bin/bash

set -e

echo "=========================================="
echo "🚀 DiscAR Installation Script"
echo "=========================================="
echo ""

if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found"
    exit 1
fi

echo "📦 Step 1/4: Creating Python 3.12 environment..."
conda create -n DiscAR python=3.12 -y

echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate DiscAR

echo ""
echo "=========================================="
echo "🔧 Step 2/4: Installing xformers"
echo "=========================================="
pip install xformers==0.0.32.post2

echo ""
echo "✅ Verifying PyTorch installation..."
python -c "import torch; print('   PyTorch:', torch.__version__); print('   CUDA:', torch.cuda.is_available())"
python -c "import xformers; print('   xformers:', xformers.__version__)"

echo ""
echo "=========================================="
echo "📚 Step 3/4: Installing other dependencies"
echo "=========================================="
conda env update -f environment.yaml

echo ""
echo "=========================================="
echo "🧠 Step 4/4: Installing tensorflow-gpu with conda for evaluator"
echo "=========================================="
echo "Installing TensorFlow with CUDA 12.8 support..."
conda install -c conda-forge tensorflow-gpu=2.19.1 -y

echo ""
echo "✅ Verifying TensorFlow installation..."
python -c "import tensorflow as tf; print('   TensorFlow:', tf.__version__)"

echo ""
echo "=========================================="
echo "✅ Verifying complete installation"
echo "=========================================="

if [ -f "test_installation.py" ]; then
    python test_installation.py
else
    echo "⚠️  test_installation.py not found"
fi

echo ""
echo "=========================================="
echo "🎉 Installation complete!"
echo "=========================================="
echo ""
echo "Activate environment: conda activate DiscAR"
echo "Run training: python train.py --config=configs/CIFAR10_VQ_ae.yaml"
echo ""
