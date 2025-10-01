#!/usr/bin/env bash
set -euo pipefail
mkdir -p models/gguf/{2bit,3bit,4bit,5bit,6bit,8bit}

download() {
  local url="$1" out="$2"
  echo "→ $out"
  curl -L -C - -o "$out" "$url"
}

download "https://huggingface.co/tensorblock/TowerInstruct-Mistral-7B-v0.2-GGUF/resolve/main/TowerInstruct-Mistral-7B-v0.2-Q2_K.gguf?download=true" \
         "models/gguf/2bit/TowerInstruct-Mistral-7B-v0.2-Q2_K.gguf"

download "https://huggingface.co/tensorblock/TowerInstruct-Mistral-7B-v0.2-GGUF/resolve/main/TowerInstruct-Mistral-7B-v0.2-Q3_K_L.gguf?download=true" \
         "models/gguf/3bit/TowerInstruct-Mistral-7B-v0.2-Q3_K_L.gguf"

download "https://huggingface.co/tensorblock/TowerInstruct-Mistral-7B-v0.2-GGUF/resolve/main/TowerInstruct-Mistral-7B-v0.2-Q4_K_M.gguf?download=true" \
         "models/gguf/4bit/TowerInstruct-Mistral-7B-v0.2-Q4_K_M.gguf"

download "https://huggingface.co/tensorblock/TowerInstruct-Mistral-7B-v0.2-GGUF/resolve/main/TowerInstruct-Mistral-7B-v0.2-Q5_K_M.gguf?download=true" \
         "models/gguf/5bit/TowerInstruct-Mistral-7B-v0.2-Q5_K_M.gguf"

download "https://huggingface.co/tensorblock/TowerInstruct-Mistral-7B-v0.2-GGUF/resolve/main/TowerInstruct-Mistral-7B-v0.2-Q6_K.gguf?download=true" \
         "models/gguf/6bit/TowerInstruct-Mistral-7B-v0.2-Q6_K.gguf"

download "https://huggingface.co/tensorblock/TowerInstruct-Mistral-7B-v0.2-GGUF/resolve/main/TowerInstruct-Mistral-7B-v0.2-Q8_0.gguf?download=true" \
         "models/gguf/8bit/TowerInstruct-Mistral-7B-v0.2-Q8_0.gguf"

echo "Done."