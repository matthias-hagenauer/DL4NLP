import json

full_path = "data/wmt24_estimated.jsonl"
out_path = "data/wmt24_estimated_normalized.jsonl"

# Compute global min/max
scores = []
with open(full_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        scores.append(obj["difficulty_score"])
min_val, max_val = min(scores), max(scores)

# Normalize and write new file
with open(full_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
    for line in f:
        obj = json.loads(line)
        s = obj["difficulty_score"]
        obj["difficulty_score_norm"] = (s - min_val) / (max_val - min_val)
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")

