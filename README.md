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

## 📚 Related Work
- [A Survey on Model Compression for Large Language Models](https://arxiv.org/abs/2307.03172)  
- [Tower: An Open Multilingual Large Language Model for Translation-Related Tasks](https://arxiv.org/abs/2402.17733)

These resources provide background on the motivation, methods, and trade-offs in compressing LLMs.

---

## 📂 Dataset
We make use of the [ALMA dataset](https://github.com/fe1ixxu/ALMA), which includes machine translation data for multiple language pairs. (NOT TRUE: FIX) 

This dataset supports:
- Training and fine-tuning
- Knowledge distillation setups
- Evaluation of compressed models

---

## Environment setup

### Prerequisites
- **Python** 3.9–3.11 (tested on 3.10)
- **NVIDIA GPU + CUDA** (cluster provides CUDA 11.8 via modules)
- Internet access to Hugging Face for model downloads

> COMET is **optional** (only needed if you run `--eval_metrics ... comet`). See **Optional: COMET** below.

### 1. Create a Conda environment

```bash
conda create -y -n nlp python=3.10
conda activate nlp

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

# Extra runtime helpers that models/metrics often need:
pip install sentencepiece protobuf safetensors
```

### Optional: COMET metric (for --eval_metrics comet)

```bash
pip install "unbabel-comet>=2.2.6"
```

---

## Quickstart

Note: this is a subset, for full dataset use `wmt24_esa.jsonl`.

```bash
# Default
python main.py \
--data data/subset.jsonl \
--outdir runs/exp1_fp16 \
--eval_metrics chrf,bleu

# 8-bit quantization
python main.py \
--data data/subset.jsonl \
--outdir runs/exp1_bnb8 \
--model_id Unbabel/TowerInstruct-Mistral-7B-v0.2 \
--quant 8bit \
--eval_metrics chrf,bleu

# 4-bit quantization
python main.py \
--data data/subset.jsonl \
--outdir runs/exp1_bnb4 \
--model_id Unbabel/TowerInstruct-Mistral-7B-v0.2 \
--quant 4bit \
--eval_metrics chrf,bleu
```


