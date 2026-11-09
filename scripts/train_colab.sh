#!/bin/bash
# ==============================================================================
# Cheb-TinyKAN: 1-Click Automated Training Script for Google Colab GPU
# ==============================================================================
set -e

echo "🚀 [1/4] Preparing Environment & Dependencies..."
cd /content/cheb-tinykan
pip install -q wfdb torchmetrics

echo "📦 [2/4] Downloading & Linking PTB-XL v1.0.3 Dataset from AWS S3 mirror..."
mkdir -p /content/dataset
if [ ! -f /content/dataset/ptb-xl.zip ]; then
    wget -c https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip -O /content/dataset/ptb-xl.zip
fi

if [ ! -d /content/dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500 ]; then
    echo "Unzipping PTB-XL dataset (~15s)..."
    unzip -q -o /content/dataset/ptb-xl.zip -d /content/dataset/
fi

rm -rf /content/cheb-tinykan/data/ptb-xl
ln -s /content/dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 /content/cheb-tinykan/data/ptb-xl
echo "✅ PTB-XL dataset verified at data/ptb-xl"

echo "🧪 [3/4] Running Comprehensive Verification Test Suites (9/9 suites)..."
python test_local.py

echo "🔥 [4/4] Starting 60-Epoch Cheb-TinyKAN Training on GPU (T4)..."
python train.py \
    --data_dir /content/cheb-tinykan/data/ptb-xl \
    --epochs 60 \
    --batch_size 128 \
    --lr 1e-3 \
    --device cuda \
    --save_dir /content/cheb-tinykan/checkpoints

echo "🎉 [COMPLETE] Training and Evaluation Finished Successfully!"
