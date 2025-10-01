import json
import matplotlib.pyplot as plt

file_path = "data/wmt24_estimated_normalized.jsonl"  

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def plot_difficulty_distribution(data, lang_pair, bins=50):
    scores = [d["difficulty_score"] for d in data if d["lp"] == lang_pair]

    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=bins, edgecolor="black")
    plt.title(f"Difficulty Score Distribution for {lang_pair}", fontsize=14)
    plt.xlabel(f"Difficulty score\n(min: {min(scores)}, max: {max(scores)})", fontsize=12)
    plt.ylabel("Number of datapoints", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_path = f"data/plots/scores_{lang_pair}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

data = load_data(file_path)

for lp in ["en-de", "en-es", "en-zh", "en-nl"]:
    plot_difficulty_distribution(data, lp)
