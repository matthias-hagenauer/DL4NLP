import json, numpy as np

file_path = "data/wmt24_estimated.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

for lp in ["en-de", "en-es", "en-zh", "en-nl"]:
    scores = [d["difficulty_score"] for d in data if d["lp"] == lp]
    arr = np.array(scores, dtype=float)
    print(lp, "n=", len(arr),
          "min=", float(arr.min()),
          "q1=", float(np.quantile(arr, 0.25)),
          "med=", float(np.median(arr)),
          "q3=", float(np.quantile(arr, 0.75)),
          "max=", float(arr.max()),
          "mean=", float(arr.mean()),
          "std=", float(arr.std()))

import random
for lp in ["en-de", "en-es", "en-zh", "en-nl"]:
    examples = [d for d in data if d["lp"] == lp]
    print(lp, examples[0]["source"])
    print(lp, examples[0]["target"])
