#!/usr/bin/env bash
set -euo pipefail

# Create output directories
mkdir -p models/gguf/{2bit,4bit,8bit}

download() {
  local url="$1" out="$2"
  echo "→ Downloading $out"
  curl -L -C - -o "$out" "$url"
}

# 2-bit model
download \
  "https://huggingface.co/tensorblock/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q2_K.gguf?download=true" \
  "models/gguf/2bit/gemma-3-12b-it-Q2_K.gguf"

# 4-bit model
download \
  "https://huggingface.co/tensorblock/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q4_K_M.gguf?download=true" \
  "models/gguf/4bit/gemma-3-12b-it-Q4_K_M.gguf"

# 8-bit model
download \
  "https://huggingface.co/tensorblock/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q8_0.gguf?download=true" \
  "models/gguf/8bit/gemma-3-12b-it-Q8_0.gguf"

echo "Done."
