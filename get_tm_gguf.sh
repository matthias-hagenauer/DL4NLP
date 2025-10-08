#!/usr/bin/env bash
set -euo pipefail
mkdir -p models/gguf/{2bit,4bit,8bit}

download() {
  local url="$1" out="$2"
  echo "→ $out"
  curl -L -C - -o "$out" "$url"
}

# 2-bit model
download \
  "https://huggingface.co/tensorblock/TowerInstruct-Mistral-7B-v0.2-GGUF/resolve/main/TowerInstruct-Mistral-7B-v0.2-Q2_K.gguf?download=true" \
  "models/gguf/2bit/TowerInstruct-Mistral-7B-v0.2-Q2_K.gguf"

# 4-bit model
download \
  "https://huggingface.co/tensorblock/TowerInstruct-Mistral-7B-v0.2-GGUF/resolve/main/TowerInstruct-Mistral-7B-v0.2-Q4_K_M.gguf?download=true" \
  "models/gguf/4bit/TowerInstruct-Mistral-7B-v0.2-Q4_K_M.gguf"

# 8-bit model
download \
  "https://huggingface.co/tensorblock/TowerInstruct-Mistral-7B-v0.2-GGUF/resolve/main/TowerInstruct-Mistral-7B-v0.2-Q8_0.gguf?download=true" \
  "models/gguf/8bit/TowerInstruct-Mistral-7B-v0.2-Q8_0.gguf"

echo "Done."