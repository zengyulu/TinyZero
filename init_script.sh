#!/bin/bash

cd /workspace

apt update
apt install gcc vim htop -f

wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

sh cuda_12.4.0_550.54.14_linux.run

#pip install vllm==0.6.3
pip install vllm==0.6.3 torch==2.4.0 torchaudio==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu124

mkdir /workspace/Qwen2.5-7B

huggingface-cli download Qwen/Qwen2.5-7B --local-dir /workspace/Qwen2.5-7B --local-dir-use-symlinks False --resume-download

git clone https://github.com/zengyulu/TinyZero.git
cd /workspace/TinyZero

pip install -e .

pip install flash-attn --no-build-isolation wandb IPython matplotlib

sh script/train_tinyzero_h200_ppo_qwen2.5_7b.sh
