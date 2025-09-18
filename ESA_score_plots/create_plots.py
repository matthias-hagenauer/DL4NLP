import json

file_path = "wmt24_esa.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

import matplotlib.pyplot as plt
from collections import defaultdict

def plot_esa_distribution(data, lang_pair):
    # Collect ESA scores for the given language
    scores = [d["esa_score"] for d in data if d.get("langs") == lang_pair and "esa_score" in d]

    
    if not scores:
        print(f"No ESA scores found for {lang_pair}")
        return
    
    # Initialize dictionary with counts
    score_dict = {}
    for i in range(101):
        score_dict[i] = 0

    for score in scores:
      score_dict[int(score)] += 1

    # Plot the score distribution
    plt.figure(figsize=(12, 6))
    x = list(map(int, score_dict.keys()))  # Ensure x-values are integers
    y = list(score_dict.values())
    
    plt.bar(x, y, color='skyblue', edgecolor='black')
    plt.title(f"ESA Score Distribution for {lang_pair}", fontsize=16)
    plt.xlabel("ESA Score", fontsize=14)
    plt.ylabel("Number of Datapoints", fontsize=14)
    plt.xticks(range(0, 101, 10))  # show ticks every 10 points
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


plot_esa_distribution(data, "en-es")
plot_esa_distribution(data, "en-ru")
plot_esa_distribution(data, "en-zh")
