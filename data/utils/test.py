import json
import matplotlib.pyplot as plt

file_path = "data/wmt24_estimated_normalized.jsonl"  

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

data = load_data(file_path)
scores = [d["difficulty_score"] for d in data]
print(f"Difficulty scores: min {min(scores)}, max {max(scores)}")