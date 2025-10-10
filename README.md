# Model Compression for Machine Translation in Large Language Models

## 📖 Project Description
This project explores the application of **model compression techniques** for **machine translation (MT)** tasks in **ALMA large language models (LLMs)**.  
The goal is to reduce model size and computational requirements **without sacrificing translation quality**.

You will experiment with the following compression strategies:
- **Quantization** – reducing the precision of weights/activations (e.g., FP16 → INT8/INT4).
- **Pruning** – removing redundant weights or attention heads.
- **Knowledge Distillation** – training smaller models (students) using larger models (teachers).

You are free to leverage the **ALMA training data** for any purpose, including:
- Distillation
- Parameter-efficient fine-tuning (PEFT, e.g. LoRA)
- Full fine-tuning

---

## Related Work
- [A Survey on Model Compression for Large Language Models](https://arxiv.org/abs/2307.03172)  
- [Tower: An Open Multilingual Large Language Model for Translation-Related Tasks](https://arxiv.org/abs/2402.17733)
- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)
- [Estimating Machine Translation Difficulty](https://arxiv.org/abs/2508.10175)

These resources provide background on the motivation, methods, and trade-offs in compressing LLMs.

---

## Dataset

We use four language pairs of the [WMT24++ dataset](https://huggingface.co/datasets/google/wmt24pp) as our primary source of machine translation data.  
To control for translation difficulty, we process it with [translation difficulty estimation](https://github.com/zouharvi/translation-difficulty-estimation) ([paper](https://arxiv.org/abs/2508.10175)).

### Language pairs
We focus on the following pairs from WMT24++:
- **en-de**
- **en-nl**
- **nl-zh**
- **en-es**

To obtain the augmented versions of the datasets that can be found in ```DifficultyEstimation/data``` the following code is used:
### Difficulty env setup:
```bash
cd DifficultyEstimation
conda create -y -n difficulty python=3.10
source activate difficulty
python -m pip install --upgrade pip

git clone git@github.com:prosho-97/guardians-mt-eval.git
cd guardians-mt-eval
pip install -e .
```

### Running the difficulty estimation:
```bash
cd DifficultyEstimation
python add_difficulty_estimation.py --input ../data/raw_data/en-de_DE.jsonl --output ../data/difficulty_augmented_data/en-de.jsonl

python add_difficulty_estimation.py --input ../data/raw_data/en-es_MX.jsonl --output ../data/difficulty_augmented_data/en-es.jsonl

python add_difficulty_estimation.py --input ../data/raw_data/en-nl_NL.jsonl --output ../data/difficulty_augmented_data/en-nl.jsonl

python add_difficulty_estimation.py --input ../data/raw_data/en-zh_CN.jsonl --output ../data/difficulty_augmented_data/en-zh.jsonl
```

### Processed datasets
We create several JSONL datasets:
- `wmt24_estimated.jsonl` — concatenated difficulty-estimated data for the selected language pairs. (This is made by concatenating the augmented pairwise datasets by hand)  
- `wmt24_estimated_normalized.jsonl` - normalized difficulty scores ([0-1]) using minmax normalization 
- `wmt24_filtered_5.jsonl` — balanced subset with 5 examples per target language for very easy testing
- `wmt24_filtered_100.jsonl` — balanced subset with 100 examples per target language  

### Example command
You can create a subset yourself using the following command (specify `--n`):

```bash
python data/utils/filter.py --mode balanced --n 100
```

---

## Metrics

We use the [🤗 Evaluate](https://huggingface.co/docs/evaluate/index) library for automatic evaluation:

- **chrF** — character n-gram F-score (`evaluate.load("chrf")`)  
- **BLEU (corpus-level)** — classic word-based metric (`evaluate.load("bleu")`)  
- **COMET-22** — learned quality estimation model ([Unbabel/wmt22-comet-da](https://huggingface.co/Unbabel/wmt22-comet-da))  

These metrics together provide complementary insights: BLEU/chrF capture surface-level overlap, while COMET better reflects human judgments.

---

## Environment setup

### Prerequisites
- **Python** 3.9–3.11 (tested on 3.10)
- **NVIDIA GPU + CUDA** (cluster provides CUDA 11.8 via modules)
- Internet access to Hugging Face for model downloads

COMET is **optional** (only needed if you run `--eval_metrics ... comet`). See **Optional: COMET** below.

### 1. Create a Conda environment

```bash
# Create env
conda create -y -n nlp python=3.10
source activate nlp

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CUDA 11.8 build, works on Snellius A100 GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project requirements (Transformers baseline, metrics, etc.)
pip install -r requirements.txt

# Insrall COMET
pip install "unbabel-comet>=2.2.6"

# Extra helpers some models/metrics need
pip install sentencepiece protobuf safetensors

# --- GGUF backend (quantized TowerMistral) ---
# 1) Install llama.cpp Python binding (CUDA 12.2 build, GPU offload support)
pip install --upgrade --no-cache-dir \
  --index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 \
  --extra-index-url https://pypi.org/simple \
  llama-cpp-python==0.3.16

# 2) Install CUDA 12.2 runtime libraries (needed by llama-cpp wheel)
pip install "nvidia-cuda-runtime-cu12==12.2.*" "nvidia-cublas-cu12==12.2.*"

# 3) Ensure a new enough C++ runtime (libstdc++ ≥ 13)
conda install -y -c conda-forge "libstdcxx-ng>=13" "libgcc-ng>=13"

chmod +x setup_llamacpp_cuda12.sh
bash setup_llamacpp_cuda12.sh
conda deactivate && conda activate nlp
```

---

## Quickstart

First dwnload quantized Tower Mistral model (GGUF format):

```bash
chmod +x get_tm_gguf.sh
bash get_tm_gguf.sh
```

Also downlaod quantized Gemini-3 model (GGUF format):

```bash
chmod +x get_gemini_gguf.sh
bash get_gemini_gguf.sh
```

### Access & Setup for Gemma-3 (Hugging Face)

`google/gemma-3-4b-it` is a gated model, so you must authenticate before first use.
1. Accept access terms
Visit [https://huggingface.co/google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) and click “Access model” / “Accept license”.

2. Create a read token
Go to Account → Acces tokens → New token → select read scope.

3. Create a `.env` file, add the following:
```bash
HUGGINGFACE_HUB_TOKEN=hf_xxx
HF_HOME=$HOME/.cache/huggingface
HF_HUB_CACHE=$HF_HOME/hub
HF_ASSETS_CACHE=$HF_HOME/assets
```
Then run `mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_ASSETS_CACHE"`

4. Load it before running or in your jobfile: `set -a; source .env; set +a`

### Run examples — TM and Gemma-3 baselines + quantized versions

You can run models like this below.
> **Note:** The examples here use the small demo file `data/wmt24_filtered_100.jsonl`.  
> For the full dataset, replace `--data data/wmt24_filtered_100.jsonl` with:  
> `--data data/wmt24_estimated.jsonl`

Set eval metrics like `--eval_metrics chrf comet`.

---

#### TowerMistral

```bash
# Baseline HuggingFace TowerMistral
python main.py \
  --model_id TM \
  --data data/wmt24_filtered_100.jsonl

# 2-bit quantized GGUF (local if available, else HF)
python main.py \
  --model_id TM_2bit 
  --data data/wmt24_filtered_100.jsonl \
  --n_gpu_layers 40

# 4-bit quantized GGUF (with explicit local override)
python main.py \
  --model_id TM_4bit \
  --gguf_path models/gguf/4bit/TowerInstruct-Mistral-7B-v0.2-Q4_K_M.gguf \
  --data data/wmt24_filtered_100.jsonl \
  --n_gpu_layers 40

# 8-bit quantized GGUF
python main.py \
  --model_id TM_8bit \
  --data data/wmt24_filtered_100.jsonl \
  --n_gpu_layers 40
```

---

#### Gemini-3 12B Instruct

```bash
# Baseline HuggingFace Gemma-3
python main.py \
  --model_id G3 \
  --data data/wmt24_filtered_100.jsonl

# 2-bit quantized GGUF
python main.py \
  --model_id G3_2bit \
  --data data/wmt24_filtered_100.jsonl \
  --n_gpu_layers 40

# 4-bit quantized GGUF (with explicit local override)
python main.py \
  --model_id G3_4bit \
  --gguf_path models/gguf/4bit/gemma-3-4b-it-Q4_K_M.gguf \
  --data data/wmt24_filtered_100.jsonl \
  --n_gpu_layers 40

# 8-bit quantized GGUF
python main.py \
  --model_id G3_8bit \
  --data data/wmt24_filtered_100.jsonl \
  --n_gpu_layers 40
```

---
