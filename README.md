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
- [Estimating Machine Translation Difficulty](https://arxiv.org/abs/2508.10175)

These resources provide background on the motivation, methods, and trade-offs in compressing LLMs.

---

## Dataset

We use the [WMT24++ dataset](https://huggingface.co/datasets/google/wmt24pp) as our primary source of machine translation data.  
To control for translation difficulty, we process it with [translation difficulty estimation](https://github.com/zouharvi/translation-difficulty-estimation) ([paper](https://arxiv.org/abs/2508.10175)).

### Language pairs
We focus on the following pairs from WMT24++:
- **en-de**
- **en-nl**
- **nl-zh**
- **en-es**

### Processed datasets
We generate several JSONL datasets:
- `wmt24_estimated.jsonl` — difficulty-estimated data for selected language pairs  
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

### Optional: COMET metric (for --eval_metrics comet)

```bash
pip install "unbabel-comet>=2.2.6"
```

---

## Quickstart

### Download quantized Tower Mistral model (GGUF format)

```bash
chmod +x get_tm_gguf.sh
bash get_tm_gguf.sh
```

### Run examples TM baseline + quantized versions

You can run models like this below.
> **Note:** The examples here use the small demo file `data/wmt24_filtered_100.jsonl`.  
> For the full dataset, replace `--data data/wmt24_filtered_100.jsonl` with:  
> `--data data/wmt24_estimated.jsonl`

Set eval metrics like `--eval_metrics chrf bleu comet`.

```bash
# Baseline HuggingFace TowerMistral
python main.py --model_id TM --data data/wmt24_filtered_100.jsonl

# 2-bit quantized GGUF (local if available, else HF)
python main.py --model_id TM_2bit --data data/wmt24_filtered_100.jsonl --n_gpu_layers 40

# 3-bit quantized GGUF
python main.py --model_id TM_3bit --data data/wmt24_filtered_100.jsonl --n_gpu_layers 40

# 4-bit quantized GGUF (with explicit local override)
python main.py --model_id TM_4bit --gguf_path models/gguf/4bit/TowerInstruct-Mistral-7B-v0.2-Q4_K_M.gguf --data data/wmt24_filtered_100.jsonl --n_gpu_layers 40

# 5-bit quantized GGUF
python main.py --model_id TM_5bit --data data/wmt24_filtered_100.jsonl --n_gpu_layers 40

# 6-bit quantized GGUF
python main.py --model_id TM_6bit --data data/wmt24_filtered_100.jsonl --n_gpu_layers 40

# 8-bit quantized GGUF
python main.py --model_id TM_8bit --data data/wmt24_filtered_100.jsonl --n_gpu_layers 40
```

# Bin-wise Translation Difficulty Analysis

This script analyzes machine translation predictions **per difficulty bin** using your existing `predictions.jsonl` files (one per model). It computes in-bin statistics (lengths, punctuation, sentence counts), aggregates metric means (e.g., chrF, COMET), and reports correlations between `difficulty_score` and lengths/metrics. Results are written **next to each model’s `predictions.jsonl`**.

---

## What it does

For **each** model folder, it will produce:

- `bin_summary.csv` — one row per `difficulty_bin`, including:
  - `count`, mean/median/std of `difficulty_score`
  - mean/std tokens & chars for `src`, `ref`, `pred`
  - punctuation density and sentence-count means
  - `exact_match_rate` (pred == ref)
  - mean/std for any metrics present in `metrics` (e.g., `chrf_mean`, `comet_seg_mean`)
- `metric_correlations.csv` — Pearson correlations within the bin between `difficulty_score` and:
  - token counts (`src/ref/pred`)
  - each metric in `metrics` (e.g., `chrf`, `comet_seg`)
- `langpair_breakdown.csv` — counts per `difficulty_bin × langpair`

## How to run

```bash

# Run for all models at once (point to the parent folder):
python analyze_bin.py results/

Or run for specific files:
python analyze_bin.py \
  results/TM/predictions.jsonl \
  results/TM_2bit/predictions.jsonl \
  results/TM_3bit/predictions.jsonl \
  results/TM_4bit/predictions.jsonl \
  results/TM_5bit/predictions.jsonl \
  results/TM_6bit/predictions.jsonl \
  results/TM_8bit/predictions.jsonl
```

## Visualization

To visualize the bin-wise statistics, you first need to install the required plotting libraries:

```bash
pip install seaborn altair altair_saver vl-convert-python statsmodels
```

To Run:

```bash
python plot_in_bin_statistics.py results/
```