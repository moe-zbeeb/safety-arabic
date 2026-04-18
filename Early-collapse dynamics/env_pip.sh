#!/usr/bin/env bash
#chmod +x env_pip.sh
#./env_pip.sh
#source /workspace/hf_env/bin/activate
#pip install trl tensorboard protobuf

set -e

ENV_NAME="hf_env"
PYTHON_VERSION="3.11"
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu124"

echo "Checking Python version"
python3 --version

echo "Creating virtual environment: $ENV_NAME"
python3 -m venv "$ENV_NAME"

echo "Activating virtual environment"
source "$ENV_NAME/bin/activate"

echo "Upgrading pip"
pip install --upgrade pip setuptools wheel

echo "Installing PyTorch with CUDA 12.4"
pip install torch torchvision torchaudio --index-url "$TORCH_CUDA_INDEX"

echo "Verifying PyTorch & CUDA"
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY

echo "Installing Hugging Face ecosystem"
pip install \
  transformers \
  datasets \
  accelerate \
  evaluate \
  peft \
  sentencepiece \
  safetensors \
  huggingface_hub \
  bitsandbytes \
  tiktoken \
  einops

echo "Installing vLLM"
pip install vllm

echo "Final verification"
python - <<'PY'
import torch, transformers, vllm

print("CUDA available:", torch.cuda.is_available())
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("vLLM:", vllm.__version__)

model = transformers.AutoModel.from_pretrained("bert-base-uncased")
print("Loaded:", model.__class__.__name__)
PY

echo "Environment setup complete ✅"