## Quickstart

```bash
python main.py \
--data data/pairs.jsonl \
--outdir runs/exp1 \
--model_id Unbabel/TowerInstruct-Mistral-7B-v0.2 \
--quant none \
--eval_metrics chrf \
--bins "0-30,30-60,60-100"

# 8-bit
python main.py --data data/pairs.jsonl --outdir runs/exp1_bnb8 \
--model_id Unbabel/TowerInstruct-Mistral-7B-v0.2 --quant 8bit --eval_metrics chrf


# 4-bit
python main.py --data data/pairs.jsonl --outdir runs/exp1_bnb4 \
--model_id Unbabel/TowerInstruct-Mistral-7B-v0.2 --quant 4bit --eval_metrics chrf
```