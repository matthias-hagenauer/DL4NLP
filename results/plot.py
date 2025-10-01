#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PREFERRED_BIN_ORDER = ["(0-82]", "(82-98]", "(98-100]", "UNBINNED"]
BIN_TO_DIFFICULTY = {
    "(0-82]": "Hard",
    "(82-98]": "Medium",
    "(98-100]": "Easy",
    "UNBINNED": "UNBINNED",
}
BIN_LABEL_MAP = {
    "(0-82]": "Hard (0–82]",
    "(82-98]": "Medium (82–98]",
    "(98-100]": "Easy (98–100]",
    "UNBINNED": "UNBINNED",
}

# ---------- Loading ----------
def resolve_summary_path(model_arg: str, base: Optional[Path]) -> Tuple[str, Path]:
    cand = Path(model_arg)
    if cand.suffix == ".json" and cand.exists():
        return (cand.parent.name or cand.stem, cand)
    if cand.is_dir() and (cand / "summary.json").exists():
        return (cand.name, cand / "summary.json")
    if base is not None:
        d = base / model_arg
        if d.is_dir() and (d / "summary.json").exists():
            return (d.name, d / "summary.json")
        j = base / model_arg
        if j.suffix == ".json" and j.exists():
            return (j.parent.name or j.stem, j)
    raise FileNotFoundError(
        f"Could not resolve summary for '{model_arg}'. "
        f"Tried '{cand}', '{cand}/summary.json'"
        + (f", '{base/model_arg}', '{base/model_arg}/summary.json'" if base else "")
    )

def load_chrf_by_bin(summary_path: Path) -> Dict[str, Dict[str, float]]:
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        return data["metrics"]["chrf"]["by_bin"]  # {bin: {"count":..,"mean":..}}
    except Exception as e:
        raise KeyError(f"{summary_path}: missing metrics.chrf.by_bin") from e

def assign_model_colors(models: List[str]) -> Dict[str, tuple]:
    cmap = plt.get_cmap("tab10")
    return {m: cmap(i % 10) for i, m in enumerate(models)}

# ---------- Plot 1: chrF per bin per model (color = model) ----------
def plot_chrf_by_bin_grouped(df: pd.DataFrame, model_order: List[str], outpath: Path) -> None:
    present = df["bin"].unique().tolist()
    bin_order = [b for b in PREFERRED_BIN_ORDER if b in present] + [b for b in present if b not in PREFERRED_BIN_ORDER]
    df["bin"] = pd.Categorical(df["bin"], categories=bin_order, ordered=True)
    df = df.sort_values(["bin", "model"])

    bins = df["bin"].cat.categories.tolist()
    models = [m for m in model_order if m in df["model"].unique()]
    colors = assign_model_colors(models)

    x = np.arange(len(bins))
    total_width = 0.8
    bar_width = total_width / max(len(models), 1)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=(10, 6))
    for j, model in enumerate(models):
        sub = df[df["model"] == model]
        means = [sub.loc[sub["bin"] == b, "mean"].iloc[0] if not sub.loc[sub["bin"] == b].empty else np.nan for b in bins]
        ax.bar(
            x + offsets[j],
            means,
            width=bar_width,
            color=colors[model],
            edgecolor="black",
            linewidth=0.5,
            label=model,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([BIN_LABEL_MAP.get(b, b) for b in bins], rotation=0)
    ax.set_title("chrF by bin (color = model)")
    ax.set_ylabel("chrF (mean)")
    ax.set_xlabel("Bin (difficulty)")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1.0), loc="upper left")

    # Annotate values
    ymax = 0.0
    for container in ax.containers:
        for bar in container:
            h = bar.get_height()
            if pd.notna(h):
                ymax = max(ymax, h)
                ax.annotate(f"{h:.2f}",
                            (bar.get_x() + bar.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=8)
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.12)

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"[OK] {outpath}")

# ---------- Plot 2: Δ chrF vs baseline per difficulty (color = model) ----------
def plot_delta_vs_baseline(df: pd.DataFrame, baseline_label: str, outpath: Path) -> None:
    """
    Δ = model_mean - baseline_mean (positive = improvement, negative = degradation).
    Only for Hard/Medium/Easy; UNBINNED excluded in this plot.
    """
    piv = df.pivot_table(index="bin", columns="model", values="mean")
    bins_of_interest = ["(0-82]", "(82-98]", "(98-100]"]

    if baseline_label not in piv.columns:
        raise ValueError(f"Baseline '{baseline_label}' not found in models: {list(piv.columns)}")

    models = [c for c in piv.columns if c != baseline_label]
    if not models:
        print("[WARN] Only baseline provided; no delta plot produced.", file=sys.stderr)
        return

    rows = []
    for b in bins_of_interest:
        if b not in piv.index:
            continue
        base_val = piv.loc[b, baseline_label]
        for m in models:
            val = piv.loc[b, m] if m in piv.columns else np.nan
            delta = val - base_val
            rows.append({"difficulty": BIN_TO_DIFFICULTY[b], "model": m, "delta": delta})

    if not rows:
        print("[WARN] No overlapping bins for delta plot.", file=sys.stderr)
        return

    ddf = pd.DataFrame(rows)
    diff_order = ["Hard", "Medium", "Easy"]
    ddf["difficulty"] = pd.Categorical(ddf["difficulty"], categories=diff_order, ordered=True)
    ddf = ddf.sort_values(["difficulty", "model"])

    # Manual grouped bars to enforce consistent colors per model
    models_order = sorted(ddf["model"].unique().tolist(), key=lambda x: x)
    colors = assign_model_colors(models_order)
    diffs = diff_order
    x = np.arange(len(diffs))
    total_width = 0.8
    bar_width = total_width / max(len(models_order), 1)
    offsets = (np.arange(len(models_order)) - (len(models_order) - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=(9, 5))
    for j, m in enumerate(models_order):
        sub = ddf[ddf["model"] == m]
        y = [sub.loc[sub["difficulty"] == d, "delta"].iloc[0] if not sub.loc[sub["difficulty"] == d].empty else np.nan for d in diffs]
        ax.bar(x + offsets[j], y, width=bar_width, color=colors[m], edgecolor="black", linewidth=0.5, label=m)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(diffs)
    ax.set_ylabel("Δ chrF (model − baseline)")
    ax.set_xlabel(f"Difficulty (baseline = {baseline_label})")
    ax.set_title("Quantization effect per difficulty (color = model)")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1.0), loc="upper left")

    # Annotate
    vals = ddf["delta"].dropna().values
    ymax = vals.max() if vals.size else 0.0
    ymin = vals.min() if vals.size else 0.0
    for container in ax.containers:
        for bar in container:
            h = bar.get_height()
            if pd.isna(h):
                continue
            ax.annotate(f"{h:+.2f}",
                        (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom" if h >= 0 else "top", fontsize=8)
    pad = (ymax - ymin) * 0.12 if ymax != ymin else 0.5
    ax.set_ylim(ymin - pad, ymax + pad)

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"[OK] {outpath}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["TM", "TM_4bit", "TM_8bit"],
                    help="Model dirs or summary.json files.")
    ap.add_argument("--base", default=None,
                    help="Optional base directory to resolve bare names (e.g., 'results').")
    ap.add_argument("--baseline", default=None,
                    help="Baseline model label (defaults to 'TM' if present, else first loaded).")
    ap.add_argument("--outdir", default="results/plots",
                    help="Directory where figures will be saved.")
    args = ap.parse_args()

    base = Path(args.base).resolve() if args.base else None
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load summaries
    rows = []
    model_labels: List[str] = []
    print("Loading chrF-by-bin from summaries:")
    for m in args.models:
        try:
            label, path = resolve_summary_path(m, base)
            by_bin = load_chrf_by_bin(path)
            for b, vals in by_bin.items():
                rows.append({
                    "model": label,
                    "bin": b,
                    "mean": float(vals["mean"]),
                    "difficulty": BIN_TO_DIFFICULTY.get(b, "UNBINNED"),
                })
            if label not in model_labels:
                model_labels.append(label)
            print(f"  - {label}: {path}  (bins={len(by_bin)})")
        except Exception as e:
            print(f"[WARN] Skipping '{m}': {e}", file=sys.stderr)

    if not rows:
        print("No valid by-bin data found. Nothing to plot.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Decide baseline
    baseline_label = args.baseline
    if baseline_label is None:
        baseline_label = "TM" if "TM" in model_labels else model_labels[0]
    if baseline_label not in model_labels:
        print(f"[WARN] Baseline '{baseline_label}' not among loaded models {model_labels}. Using first.", file=sys.stderr)
        baseline_label = model_labels[0]
    print(f"Using baseline: {baseline_label}")

    # (1) chrF by bin (color = model)
    plot_chrf_by_bin_grouped(df.copy(), model_labels, outdir / "chrf_by_bin_all_models.png")

    # (2) Δ chrF vs baseline per difficulty (color = model)
    plot_delta_vs_baseline(df.copy(), baseline_label, outdir / "chrf_delta_vs_baseline_by_difficulty.png")


if __name__ == "__main__":
    main()
