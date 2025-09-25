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