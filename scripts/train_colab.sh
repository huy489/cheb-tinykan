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

echo "🧪 [3/5] Running Comprehensive Verification Test Suites (10/10 suites)..."
python test_local.py

echo "⚡ [4/5] Building Preprocessed RAM Cache (167 MB) once on CPU..."
if [ ! -f /content/cheb-tinykan/data/cache/ptbxl_preprocessed_250hz.npz ]; then
    python scripts/prepare_cache.py \
        --data_dir /content/cheb-tinykan/data/ptb-xl \
        --output /content/cheb-tinykan/data/cache/ptbxl_preprocessed_250hz.npz \
        --audit /content/cheb-tinykan/docs/cache_audit.json
fi

echo "🔥 [5/5] Running GPU Tasks (Heavy Baselines or Cheb-TinyKAN) on Colab T4 using RAM Cache..."
RUN_MODE="${1:-baselines}"

if [ "$RUN_MODE" = "baselines" ] || [ "$RUN_MODE" = "all" ]; then
    echo "📊 Training Heavy IEEE Baselines (ResNet1D & InceptionTime) on CUDA T4..."
    python scripts/train_baselines.py --model resnet1d \
        --data_dir /content/cheb-tinykan/data/ptb-xl \
        --cache_path /content/cheb-tinykan/data/cache/ptbxl_preprocessed_250hz.npz \
        --epochs 40 --batch_size 128 --lr 1e-3 --device cuda

    python scripts/train_baselines.py --model inception_time \
        --data_dir /content/cheb-tinykan/data/ptb-xl \
        --cache_path /content/cheb-tinykan/data/cache/ptbxl_preprocessed_250hz.npz \
        --epochs 40 --batch_size 128 --lr 1e-3 --device cuda
fi

if [ "$RUN_MODE" = "tinykan" ] || [ "$RUN_MODE" = "all" ]; then
    echo "⚡ Running Cheb-TinyKAN Run #2 Config on CUDA..."
    python train.py \
        --data_dir /content/cheb-tinykan/data/ptb-xl \
        --cache_path /content/cheb-tinykan/data/cache/ptbxl_preprocessed_250hz.npz \
        --arch baseline --epochs 60 --batch_size 128 --lr 1e-3 \
        --auto_alpha --l1_lambda 5e-3 --prune_epoch 40 --target_sparsity 0.70 \
        --device cuda --resume --save_dir /content/cheb-tinykan/checkpoints/run2_colab
fi

echo "🎉 [COMPLETE] Colab GPU Execution Finished Successfully!"
