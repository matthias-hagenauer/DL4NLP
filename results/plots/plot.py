import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_ROOT = "results"
MODELS = {               
    "TM": "baseline",
    "TM_2bit": "2-bit",
    "TM_3bit": "3-bit",
    "TM_4bit": "4-bit",
    "TM_5bit": "5-bit",
    "TM_8bit": "8-bit",
}
LANGPAIRS = ["en-de", "en-es", "en-zh", "en-nl"]
METRIC = "comet_seg"        
OUT_DIR = "results/plots"
COLORMAP = plt.get_cmap("Dark2")

def _bin_key(label: str) -> float:
    """Sort key: parse '(lo-hi]' by the 'lo' number; UNBINNED last."""
    s = str(label).strip()
    if s.upper() == "UNBINNED":
        return np.inf
    try:
        inner = s.strip("()[]")
        lo_s, _ = inner.split("-")
        return float(lo_s)
    except Exception:
        return np.inf

def _collapse_to_e_m_h(df_bins: pd.Series) -> dict:
    """
    Given the set of difficulty_bin strings present in a model's CSV,
    return a mapping {original_bin_str -> 'easy'|'medium'|'hard'} based on rank.
    """
    uniq = list(dict.fromkeys(df_bins.astype(str).map(str.strip)))
    uniq_sorted = sorted(uniq, key=_bin_key)  # lowest -> highest
    mapping = {}
    labels = ["easy", "medium", "hard"]
    # map lowest->easy, highest->hard, everything between->medium
    for i, b in enumerate(uniq_sorted):
        if i == 0:
            mapping[b] = "easy"
        elif i == len(uniq_sorted) - 1:
            mapping[b] = "hard"
        else:
            mapping[b] = "medium"
    return mapping

def load_model_df(model_id: str, pretty: str) -> pd.DataFrame:
    csv_path = os.path.join(RESULTS_ROOT, model_id, "rows.csv")
    df = pd.read_csv(csv_path, usecols=["langpair", "difficulty_bin", METRIC])

    # drop UNBINNED & NaNs
    df["difficulty_bin"] = df["difficulty_bin"].astype(str).str.strip()
    df = df[df["difficulty_bin"].str.upper() != "UNBINNED"].copy()
    df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")
    df = df.dropna(subset=[METRIC])

    # collapse to easy/medium/hard per model based on that model's bins
    mapping = _collapse_to_e_m_h(df["difficulty_bin"])
    df["difficulty"] = df["difficulty_bin"].map(mapping)

    # mean metric per langpair × (easy/medium/hard)
    g = (df.groupby(["langpair", "difficulty"], as_index=False)[METRIC]
            .mean())
    g["model"] = pretty
    return g

def plot_langpair(model_tables: dict, langpair: str, out_dir: str):
    # build matrix rows = ['easy','medium','hard'], cols = models
    levels = ["easy", "medium", "hard"]
    models = list(model_tables.keys())
    Y = np.full((len(levels), len(models)), np.nan)

    for j, m in enumerate(models):
        sub = model_tables[m]
        row = sub[sub["langpair"] == langpair].set_index("difficulty")
        for i, lev in enumerate(levels):
            if lev in row.index:
                Y[i, j] = row.at[lev, METRIC]

    x = np.arange(len(levels))
    width = 0.8 / max(len(models), 1)

    plt.figure(figsize=(7, 4))
    for j, m in enumerate(models):
        color = COLORMAP(j % COLORMAP.N) 
        plt.bar(
            x + (j - (len(models)-1)/2) * width,
            Y[:, j],
            width=width,
            label=m,
            color=color
        )

    plt.xticks(x, levels)
    plt.ylabel(METRIC)
    plt.ylim(0, 1)  # force scale from 0 to 1
    plt.title(f"{langpair} · {METRIC} vs difficulty")
    plt.legend(loc="upper right")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{METRIC}_{langpair}.png"))
    plt.close()


def main():
    # load & aggregate each model
    model_tables = {}
    for model_id, pretty in MODELS.items():
        g = load_model_df(model_id, pretty)
        model_tables[pretty] = g

    # plot per langpair
    for lp in LANGPAIRS:
        plot_langpair(model_tables, lp, OUT_DIR)

if __name__ == "__main__":
    main()
