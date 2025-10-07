import os
import json
import pandas as pd


##################################################
################## LOAD DATA #####################
##################################################

"""
This script reads in all predictions, and creates a big dataframe. From that it creates violin plots for each 
language pair, and model. 
The results dictionary should look like this:

results/
├── TM/
│   └── predictions.jsonl
├── TM_2bit/
│   └── predictions.jsonl
├── TM_3bit/
│   └── predictions.jsonl
...
├── TM_8bit/
│   └── predictions.jsonl

Example structure of the combined DataFrame (df):
┌────┬──────────────────────────────────────────────────────────────┬────────┬──────────┬──────────┬──────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────┬──────────────────────────────────────────┬────────────────┬───────────────────────────────────────────────┬──────────┬───────────────────────┬───────────────────────────┬───────────────────────┬────────────────┬──────────────────┬───────────────────┬──────────────────────┬──────────┬───────────┬────────────────────┐
│    │ id                                                           │ langpair │ src_lang │ tgt_lang │ src                                                          │ ref                                      │ pred                                     │ difficulty_score │ difficulty_bin                              │ meta_domain │ meta_segment_id │ meta_is_bad_source │ meta_original_target                      │ meta_difficulty_score │ metrics_chrf │ metrics_comet_seg │ binning           │ model │ bin_lower │ difficulty_label │
├────┼──────────────────────────────────────────────────────────────┼────────┼──────────┼──────────┼──────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────┼────────────────┼───────────────────────────────────────────────┼──────────┼───────────────────────┼───────────────────────────┼───────────────────────┼────────────────┼──────────────────┼───────────────────┼──────────────────────┼──────────┼───────────┼────────────────────┤
│ 0  │ test-en-social_112289379466442912:654:en-de                 │ en-de  │ en       │ de       │ Does this make sense? Would this be useful?                     │ Macht das Sinn? Wäre das hilfreich?      │ Macht das Sinn? Wäre das nützlich?      │ 0.978634          │ (0.9307763735422399-0.9958426710938243]     │ social     │ 654               │ False             │ Macht das Sinn? Wäre das hilfreich?   │ 0.978634                  │ 70.314645     │ 0.976394          │ quantile_balanced │ TM   │ 0.930776  │ Hard             │
│ 1  │ test-en-social_112112980319428992:418:en-zh                 │ en-zh  │ en       │ zh       │ I'm splurging on a new set of frames, these re...              │ 我花大价钱买了一副新镜框，这副红色的我很喜欢。 │ 我准备买一副新眼镜，这副红色的我特别喜欢。        │ 0.889706          │ (0.8641313217143293-0.9307763735422399]     │ social     │ 418               │ False             │ 我花高价买了一副新镜框，是我非常喜欢的红色.  │ 0.889706                  │ 33.162672     │ 0.881692          │ quantile_balanced │ TM   │ 0.864131  │ Medium           │
│ 2  │ test-en-social_112122127346453600:459:en-nl                 │ en-nl  │ en       │ nl       │ Chapter five is a long one, but so fucking ang...              │ Hoofdstuk vijf wordt lang, maar echt vres │ Hoofdstuk vijf is lang, maar zo ontzette│ 0.886705          │ (0.8641313217143293-0.9307763735422399]     │ social     │ 459               │ False             │ Hoofstuk vijf wordt een lange, maar ec│ 0.886705                  │ 40.075556     │ 0.723553          │ quantile_balanced │ TM   │ 0.864131  │ Medium           │
└────┴──────────────────────────────────────────────────────────────┴────────┴──────────┴──────────┴──────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────┴──────────────────────────────────────────┴────────────────┴───────────────────────────────────────────────┴──────────┴───────────────────────┴───────────────────────────┴───────────────────────┴────────────────┴──────────────────┴───────────────────┴──────────────────────┴──────────┴───────────┴────────────────────┘

"""

# Path to your results directory
base_dir = "../../results"

# Helper: flatten nested dictionaries (1-level deep)
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Collect all prediction data
all_data = []

for model_dir in os.listdir(base_dir):
    model_path = os.path.join(base_dir, model_dir)
    predictions_path = os.path.join(model_path, "predictions.jsonl")
    
    if os.path.isfile(predictions_path):
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                entry_flat = flatten_dict(entry)   # flatten nested dicts
                entry_flat["model"] = model_dir    # add model identifier
                all_data.append(entry_flat)

# Convert to DataFrame
df = pd.DataFrame(all_data)

print(f"✅ Loaded {len(df)} entries from {df['model'].nunique()} models.")


###################################################
############### PRE-PROCESS DATA ##################
###################################################

import re 

# --- Filter out entries without valid difficulty bins ---
df = df[df["difficulty_bin"].notna()]
df = df[df["difficulty_bin"] != ""]  # remove empty strings
df = df[df["difficulty_bin"] != "UNBINNED"]

# --- Sort bins numerically ---
# Extract lower bound from strings like "(0.9307-0.9958]"
def get_lower_bound(bin_str):
    match = re.search(r"([\d.]+)-", bin_str)
    return float(match.group(1)) if match else float("inf")

df["bin_lower"] = df["difficulty_bin"].apply(get_lower_bound)

# Sort unique bins and assign easy → hard labels
unique_bins = sorted(df["difficulty_bin"].unique(), key=lambda b: get_lower_bound(b))
labels = ["Easy", "Medium", "Hard", "Very Hard"][:len(unique_bins)]

# Map old bin names to difficulty labels
bin_map = dict(zip(unique_bins, labels))
df["difficulty_label"] = df["difficulty_bin"].map(bin_map)

print("📊 Difficulty bin mapping:")
for old, new in bin_map.items():
    print(f"  {old} → {new}")

###############################################
################## CREATE PLOTS ###############
###############################################

import seaborn as sns
import matplotlib.pyplot as plt

# Ensure difficulty order
difficulty_order = ["Easy", "Medium", "Hard"]

# Create a FacetGrid: one column per language pair, one row per model
g = sns.FacetGrid(
    df,
    row="model",
    col="langpair",
    sharey=True,
    margin_titles=True,
    despine=False,
    height=3.5,
    aspect=1.2
)

# Draw the violin plots in each facet
g.map_dataframe(
    sns.violinplot,
    x="difficulty_label",
    y="metrics_comet_seg",
    order=difficulty_order,
    inner="box",
    cut=0,
    scale="width",
    palette="Blues"
)

# Adjust layout and titles
g.set_axis_labels("Difficulty", "COMET Segment Score")
g.set_titles(row_template="{row_name}", col_template="{col_name}")
for ax in g.axes.flatten():
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.subplots_adjust(top=0.92)
g.fig.suptitle("COMET Score Distributions by Difficulty, Model, and Language Pair", fontsize=16)

plt.savefig("plots_violin/all_models_comparison_grid.png", dpi=250)
plt.show()
