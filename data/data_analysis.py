import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load JSONL file
df = pd.read_json("wmt24_estimated_normalized.jsonl", lines=True)

# Optional: filter out invalid or missing scores
df = df[df["difficulty_score"].notna()]

# Set up the figure
plt.figure(figsize=(6, 6))

# Create the violin plot for the overall distribution
sns.violinplot(
    data=df,
    y="difficulty_score",  
    inner="box",
    cut=0,
    color="skyblue"        # optional: makes it look cleaner
)

# Labels and title
plt.title("Distribution of Difficulty Scores (Entire Dataset)", fontsize=14)
plt.ylabel("Difficulty Score", fontsize=12)
plt.xlabel("")  # no need for x-label since there’s just one
plt.tight_layout()

# --- Save the plot ---
plt.savefig("difficulty_distribution_violin.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()
