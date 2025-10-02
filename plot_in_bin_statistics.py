#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot in-bin statistics for each model.

- Enforces three difficulty bins and renames them to: easy / medium / hard.
- Adds Seaborn-based visuals:
  * Smoothed (LOWESS) metric vs difficulty scatter (per-sentence)
  * Joint scatter + marginal histograms (per-sentence)
  * KDEs of token lengths per bin
  * Pred/Ref length ratio histograms per bin
  * Number-consistency heuristic (digits preserved from src in pred) per bin
- Adds optional Altair PNGs (saved via altair_saver), e.g., layered scatter with tooltips (saved as PNG).
- Keeps your previous PNGs (lengths, exact-match absolute counts, per-metric means, corr heatmap) and combined PDF report.

Usage
-----
# Plot all models under results/
python plot_in_bin_statistics.py results/

# Single model
python plot_in_bin_statistics.py results/TM_8bit/in_bin_statistics/

# Multiple model stat dirs
python plot_in_bin_statistics.py results/TM/in_bin_statistics/ results/TM_2bit/in_bin_statistics/
"""

import argparse
from pathlib import Path
import json
import math
import re
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---- Optional libs ----
try:
    import seaborn as sns
    _HAVE_SEABORN = True
except Exception:
    _HAVE_SEABORN = False

try:
    import statsmodels  # noqa: F401  # presence enables LOWESS in seaborn.regplot
    _HAVE_STATSMODELS = True
except Exception:
    _HAVE_STATSMODELS = False

# Altair (PNG export via altair_saver + vl-convert)
try:
    import altair as alt
    from altair_saver import save as alt_save
    _HAVE_ALTAIR = True
except Exception:
    _HAVE_ALTAIR = False

# ---------------------- bin mapping ----------------------
BIN_LABELS_MAP = {
    "(0.0-0.8641313217143293]": "easy",
    "(0.8641313217143293-0.9307763735422399]": "medium",
    "(0.9307763735422399-0.9958426710938243]": "hard",
}
BIN_ORDER = ["easy", "medium", "hard"]

def map_and_filter_bins(df: pd.DataFrame, col="difficulty_bin") -> pd.DataFrame:
    """Keep only the three defined bins and rename to easy/medium/hard."""
    if col not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["_bin_label"] = d[col].map(BIN_LABELS_MAP)
    d = d[~d["_bin_label"].isna()].copy()
    d["_bin_order"] = d["_bin_label"].apply(lambda x: BIN_ORDER.index(x) if x in BIN_ORDER else 999)
    d = d.sort_values(["_bin_order", "_bin_label"]).drop(columns=["_bin_order"])
    return d

# ---------------------- helpers ----------------------
def discover_stat_dirs(inputs):
    out = []
    for p in inputs:
        pth = Path(p)
        if pth.is_dir():
            if pth.name == "in_bin_statistics":
                if (pth / "bin_summary.csv").exists():
                    out.append(pth)
                continue
            for cand in pth.rglob("in_bin_statistics"):
                if (cand / "bin_summary.csv").exists():
                    out.append(cand)
    return sorted({str(d): d for d in out}.values())

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                s2 = s.strip('"')
                try:
                    yield json.loads(s2)
                except Exception:
                    continue

def to_df_raw(preds_path: Path) -> pd.DataFrame:
    """Load per-sentence rows from predictions.jsonl beside stat_dir/.."""
    rows = []
    for r in iter_jsonl(preds_path):
        rows.append({
            "id": r.get("id"),
            "src": r.get("src", ""),
            "ref": r.get("ref", ""),
            "pred": r.get("pred", ""),
            "difficulty_score": float(r.get("difficulty_score")) if r.get("difficulty_score") is not None else math.nan,
            "difficulty_bin": r.get("difficulty_bin"),
            # metrics (dynamic)
            **{f"m__{k}": v for k, v in (r.get("metrics") or {}).items()},
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Map bins & basic features
    df["_bin_label"] = df["difficulty_bin"].map(BIN_LABELS_MAP)
    df = df[~df["_bin_label"].isna()].copy()
    # token lengths (whitespace)
    df["src_tokens"]  = df["src"].fillna("").str.split().str.len()
    df["ref_tokens"]  = df["ref"].fillna("").str.split().str.len()
    df["pred_tokens"] = df["pred"].fillna("").str.split().str.len()
    # char lengths
    df["src_chars"]  = df["src"].fillna("").str.len()
    df["ref_chars"]  = df["ref"].fillna("").str.len()
    df["pred_chars"] = df["pred"].fillna("").str.len()
    # ratios & punctuation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["len_ratio_pred_ref"] = (df["pred_tokens"] / df["ref_tokens"]).replace([math.inf, -math.inf], math.nan)
    df["src_punct_density"]  = df["src"].fillna("").str.count(r"[.,;:!?\-—–…\"'()\[\]{}]").div(df["src_chars"].clip(lower=1))
    df["pred_punct_density"] = df["pred"].fillna("").str.count(r"[.,;:!?\-—–…\"'()\[\]{}]").div(df["pred_chars"].clip(lower=1))

    # Simple numeric consistency heuristic: every digit group in src should appear in pred
    def numbers(text):
        return set(re.findall(r"\d+(?:[.,]\d+)?", text or ""))
    src_nums = df["src"].apply(numbers)
    pred_nums = df["pred"].apply(numbers)
    df["num_consistent"] = [1 if (s == set() or s.issubset(p)) else 0 for s, p in zip(src_nums, pred_nums)]

    return df

def figsize(): return (8, 5)

# ---------------------- “original” plots (bin_summary / corr) ----------------------
def plot_lengths_tokens(df_sum: pd.DataFrame, out_dir: Path, model_name: str, pdf: PdfPages):
    need = ["_bin_label", "src_tokens_mean", "ref_tokens_mean", "pred_tokens_mean"]
    if not all(c in df_sum.columns for c in need): return
    d = df_sum[need].set_index("_bin_label").reindex(BIN_ORDER)
    x = range(len(d))
    plt.figure(figsize=figsize())
    plt.plot(x, d["src_tokens_mean"], marker="o", label="src_tokens_mean")
    plt.plot(x, d["ref_tokens_mean"], marker="o", label="ref_tokens_mean")
    plt.plot(x, d["pred_tokens_mean"], marker="o", label="pred_tokens_mean")
    plt.title(f"{model_name} — Mean tokens per bin")
    plt.xlabel("bin"); plt.ylabel("tokens (mean)")
    plt.xticks(ticks=x, labels=d.index); plt.legend(); plt.tight_layout()
    fp = out_dir / "lengths_tokens.png"; plt.savefig(fp, dpi=200); pdf.savefig(); plt.close()

def plot_lengths_chars(df_sum: pd.DataFrame, out_dir: Path, model_name: str, pdf: PdfPages):
    need = ["_bin_label", "src_chars_mean", "ref_chars_mean", "pred_chars_mean"]
    if not all(c in df_sum.columns for c in need): return
    d = df_sum[need].set_index("_bin_label").reindex(BIN_ORDER)
    x = range(len(d))
    plt.figure(figsize=figsize())
    plt.plot(x, d["src_chars_mean"], marker="o", label="src_chars_mean")
    plt.plot(x, d["ref_chars_mean"], marker="o", label="ref_chars_mean")
    plt.plot(x, d["pred_chars_mean"], marker="o", label="pred_chars_mean")
    plt.title(f"{model_name} — Mean characters per bin")
    plt.xlabel("bin"); plt.ylabel("chars (mean)")
    plt.xticks(ticks=x, labels=d.index); plt.legend(); plt.tight_layout()
    fp = out_dir / "lengths_chars.png"; plt.savefig(fp, dpi=200); pdf.savefig(); plt.close()

def plot_exact_match_absolute(df_sum: pd.DataFrame, out_dir: Path, model_name: str, pdf: PdfPages):
    need = ["_bin_label", "exact_match_rate", "count"]
    if not all(c in df_sum.columns for c in need): return
    d = df_sum[need].set_index("_bin_label").reindex(BIN_ORDER)
    d["exact_matches"] = (d["exact_match_rate"] * d["count"]).round(0)
    total = int(d["count"].iloc[0]) if not d["count"].isna().all() else None
    plt.figure(figsize=figsize())
    plt.bar(d.index, d["exact_matches"])
    ttl_suffix = f" (out of {total} per bin)" if total is not None else ""
    plt.title(f"{model_name} — Exact matches per bin{ttl_suffix}")
    plt.xlabel("bin"); plt.ylabel(f"exact matches{ttl_suffix}")
    plt.tight_layout()
    fp = out_dir / "exact_match_absolute.png"; plt.savefig(fp, dpi=200); pdf.savefig(); plt.close()

def plot_metrics_means_separate(df_sum: pd.DataFrame, out_dir: Path, model_name: str, pdf: PdfPages):
    cols = [c for c in df_sum.columns if c.endswith("_mean")]
    cols = [c for c in cols if not c.startswith(("src_", "ref_", "pred_", "difficulty_"))]
    if not cols or "_bin_label" not in df_sum.columns: return
    d = df_sum[["_bin_label"] + cols].set_index("_bin_label").reindex(BIN_ORDER)
    for c in cols:
        plt.figure(figsize=figsize())
        plt.bar(d.index, d[c])
        pretty = c.replace("_mean", "")
        plt.title(f"{model_name} — {pretty} mean per bin")
        plt.xlabel("bin"); plt.ylabel(f"{pretty} (mean)")
        plt.tight_layout()
        fp = out_dir / f"metric_mean__{pretty}.png"; plt.savefig(fp, dpi=200); pdf.savefig(); plt.close()

def plot_metric_correlations_heatmap(df_corr: pd.DataFrame, out_dir: Path, model_name: str, pdf: PdfPages):
    if df_corr.empty or "difficulty_bin" not in df_corr.columns: return
    d = map_and_filter_bins(df_corr, col="difficulty_bin")
    if d.empty: return
    corr_cols = [c for c in d.columns if c.startswith("corr_diff__")]
    if not corr_cols: return
    d = d.set_index("_bin_label").reindex(BIN_ORDER)
    mat = d[corr_cols].values
    plt.figure(figsize=(max(8, 1 + 0.5 * len(corr_cols)), 4.8))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"{model_name} — Correlations w/ difficulty_score")
    plt.yticks(range(len(BIN_ORDER)), BIN_ORDER)
    xlabels = [c.replace("corr_diff__", "") for c in corr_cols]
    plt.xticks(range(len(corr_cols)), xlabels, rotation=30, ha="right")
    plt.tight_layout()
    fp = out_dir / "metric_correlations_heatmap.png"; plt.savefig(fp, dpi=200); pdf.savefig(); plt.close()

# ---------------------- NEW: per-sentence (Seaborn) ----------------------
def sns_smooth_metric_vs_difficulty(df_raw: pd.DataFrame, metric: str, out_dir: Path, model_name: str, pdf: PdfPages):
    if not _HAVE_SEABORN or metric not in df_raw.columns: return
    plt.figure(figsize=(8, 5))
    # LOWESS only if statsmodels present
    lowess_on = _HAVE_STATSMODELS
    sns.regplot(
        data=df_raw, x="difficulty_score", y=metric,
        lowess=lowess_on, scatter_kws={"s": 10, "alpha": 0.4}, line_kws={"lw": 2}
    )
    plt.title(f"{model_name} — {metric.replace('m__','')} vs difficulty (LOWESS={lowess_on})")
    plt.xlabel("difficulty_score"); plt.ylabel(metric.replace("m__",""))
    plt.tight_layout()
    fp = out_dir / f"smooth_{metric}_vs_difficulty.png"; plt.savefig(fp, dpi=200); pdf.savefig(); plt.close()

def sns_joint_scatter(df_raw: pd.DataFrame, x: str, y: str, out_dir: Path, model_name: str, pdf: PdfPages):
    if not _HAVE_SEABORN or y not in df_raw.columns: return
    g = sns.jointplot(data=df_raw, x=x, y=y, kind="reg", height=5)
    g.fig.suptitle(f"{model_name} — joint {y.replace('m__','')} vs {x}", y=1.02)
    fp = out_dir / f"joint_{y}_vs_{x}.png"; g.savefig(fp, dpi=200); pdf.savefig(g.fig); plt.close(g.fig)

def sns_kde_tokens(df_raw: pd.DataFrame, out_dir: Path, model_name: str, pdf: PdfPages):
    if not _HAVE_SEABORN: return
    plt.figure(figsize=(8,5))
    for b in BIN_ORDER:
        sns.kdeplot(data=df_raw[df_raw["_bin_label"]==b], x="src_tokens", fill=True, alpha=0.3, label=f"{b} src")
    plt.title(f"{model_name} — KDE of src_tokens by bin")
    plt.xlabel("src_tokens"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
    fp = out_dir / "kde_src_tokens.png"; plt.savefig(fp, dpi=200); pdf.savefig(); plt.close()

def sns_lenratio_hist(df_raw: pd.DataFrame, out_dir: Path, model_name: str, pdf: PdfPages):
    if not _HAVE_SEABORN: return
    plt.figure(figsize=(8,5))
    for b in BIN_ORDER:
        sns.histplot(df_raw[df_raw["_bin_label"]==b]["len_ratio_pred_ref"], bins=30, stat="density", alpha=0.5, label=b)
    plt.title(f"{model_name} — pred/ref token length ratio by bin")
    plt.xlabel("len_ratio_pred_ref"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
    fp = out_dir / "hist_len_ratio_pred_ref.png"; plt.savefig(fp, dpi=200); pdf.savefig(); plt.close()

def sns_number_consistency(df_raw: pd.DataFrame, out_dir: Path, model_name: str, pdf: PdfPages):
    rate = df_raw.groupby("_bin_label")["num_consistent"].mean().reindex(BIN_ORDER)
    plt.figure(figsize=(8,5))
    plt.bar(rate.index, (rate * 100.0))
    plt.title(f"{model_name} — Number consistency (src→pred) per bin")
    plt.xlabel("bin"); plt.ylabel("consistent (%)"); plt.tight_layout()
    fp = out_dir / "number_consistency_rate.png"; plt.savefig(fp, dpi=200); pdf.savefig(); plt.close()

# ---------------------- NEW: Altair (PNG, optional) ----------------------
def altair_scatter_metric(df_raw: pd.DataFrame, metric: str, out_dir: Path, model_name: str, pdf: PdfPages):
    if not _HAVE_ALTAIR or metric not in df_raw.columns: return
    # Altair PNG saving uses altair_saver; embed minimal fields
    chart = (
        alt.Chart(df_raw).mark_circle(size=20, opacity=0.4)
        .encode(
            x=alt.X("difficulty_score:Q", title="difficulty_score"),
            y=alt.Y(f"{metric}:Q", title=metric.replace("m__","")),
            color=alt.Color("_bin_label:N", sort=BIN_ORDER),
        )
        .properties(width=500, height=320, title=f"{model_name} — {metric.replace('m__','')} vs difficulty")
    )
    png_path = out_dir / f"altair_scatter_{metric}.png"
    try:
        alt_save(chart, str(png_path))
    except Exception:
        # If saver not configured, skip silently
        return
    # Also drop a static image into the PDF
    img = plt.imread(png_path)
    plt.figure(figsize=(8,5)); plt.imshow(img); plt.axis("off"); pdf.savefig(); plt.close()

# ---------------------- per model ----------------------
def plot_one_model(stat_dir: Path):
    model_name = stat_dir.parent.name  # e.g., TM_8bit

    # summaries
    df_sum  = safe_read_csv(stat_dir / "bin_summary.csv")
    df_corr = safe_read_csv(stat_dir / "metric_correlations.csv")
    df_sum  = map_and_filter_bins(df_sum, col="difficulty_bin")

    # raw per-sentence
    preds_path = stat_dir.parent / "predictions.jsonl"
    df_raw = to_df_raw(preds_path)
    # which sentence-level metrics do we have?
    sent_metrics = [c for c in df_raw.columns if c.startswith("m__")]

    if df_sum.empty and df_raw.empty and df_corr.empty:
        print(f"[skip] No matching data in {stat_dir}")
        return

    report_pdf = stat_dir / "in_bin_statistics_report.pdf"
    with PdfPages(report_pdf) as pdf:
        # Original bin-summary based plots
        if not df_sum.empty:
            plot_lengths_tokens(df_sum, stat_dir, model_name, pdf)
            plot_lengths_chars(df_sum, stat_dir, model_name, pdf)
            plot_exact_match_absolute(df_sum, stat_dir, model_name, pdf)
            plot_metrics_means_separate(df_sum, stat_dir, model_name, pdf)
            plot_metric_correlations_heatmap(df_corr, stat_dir, model_name, pdf)

        # Advanced per-sentence plots (Seaborn)
        if not df_raw.empty:
            if _HAVE_SEABORN:
                # Smoothed difficulty vs metric for each available metric
                for m in sent_metrics:
                    sns_smooth_metric_vs_difficulty(df_raw, m, stat_dir, model_name, pdf)
                    sns_joint_scatter(df_raw, "difficulty_score", m, stat_dir, model_name, pdf)
                sns_kde_tokens(df_raw, stat_dir, model_name, pdf)
                sns_lenratio_hist(df_raw, stat_dir, model_name, pdf)
                sns_number_consistency(df_raw, stat_dir, model_name, pdf)

            # Optional Altair PNGs for the first couple of metrics
            if _HAVE_ALTAIR and sent_metrics:
                for m in sent_metrics[:2]:
                    altair_scatter_metric(df_raw, m, stat_dir, model_name, pdf)

    print(f"Wrote plots to: {stat_dir}")
    print(f" - {report_pdf.name} (combined PDF)")

# ---------------------- Cross-model grouped plots from summary.json ----------------------

def _common_base_dir(stat_dirs):
    """Find a sensible common parent to store combined plots."""
    paths = [Path(d).resolve() for d in stat_dirs]
    if not paths:
        return Path("results").resolve()
    # Compute common prefix of all stat_dir paths, then go one level up if the last part is 'in_bin_statistics'
    common_parts = list(paths[0].parts)
    for p in paths[1:]:
        # trim to shared prefix
        i = 0
        while i < min(len(common_parts), len(p.parts)) and common_parts[i] == p.parts[i]:
            i += 1
        common_parts = common_parts[:i]
    common = Path(*common_parts)
    # If the common path is the stats folder itself, step up to the model parent
    if common.name == "in_bin_statistics":
        return common.parent
    return common if common.parts else Path("results").resolve()


def _model_order_key(name: str) -> int:
    """
    Order models as: TM, TM_2bit, TM_3bit, TM_4bit, TM_5bit, TM_6bit, TM_8bit (or similar).
    """
    name_l = name.lower()
    if name_l in {"tm", "towermistral"}:
        return 0
    m = re.search(r"(\d+)bit", name_l)
    if m:
        val = int(m.group(1))
        order_map = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6}
        return order_map.get(val, 99)
    return 98  # unknowns near end


def collect_metric_by_bin(stat_dirs, metric_name: str) -> pd.DataFrame:
    """
    Read summary.json from each model directory.
    Return DataFrame with columns: model, langpair, bin_label (easy/medium/hard), <metric_name>_mean
    Expects: summary.json -> metrics -> <metric_name> -> by_langpair_bin -> "en-de | (a-b]" : {count, mean}
    """
    rows = []
    for sd in stat_dirs:
        model_dir = Path(sd).parent
        model_name = model_dir.name
        summary_path = model_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        metrics_blob = (data.get("metrics") or {}).get(metric_name, {})
        by_lp_bin = metrics_blob.get("by_langpair_bin", {})
        if not by_lp_bin:
            # fall back to top-level (older format), e.g., data.get(metric_name, {})
            by_lp_bin = (data.get(metric_name, {}) or {}).get("by_langpair_bin", {})

        for key, obj in by_lp_bin.items():
            if " | " not in key:
                continue
            lp, bin_str = key.split(" | ", 1)
            label = BIN_LABELS_MAP.get(bin_str)
            if not label:
                continue  # skip UNBINNED/unknown bins
            mean = obj.get("mean", None)
            if mean is None:
                continue
            rows.append({
                "model": model_name,
                "langpair": lp,
                "bin_label": label,
                f"{metric_name}_mean": float(mean),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def plot_cross_model_metric(df_all: pd.DataFrame, metric_name: str, out_base: Path):
    """
    Make one grouped bar plot per language pair for a given metric.
    Saves PNGs under: <out_base>/in_bin_statistics_across_models/<metric_name>/
    """
    if df_all.empty:
        print(f"[cross-model] No {metric_name.upper()}-by-bin data found.")
        return

    out_dir = out_base / "in_bin_statistics_across_models" / metric_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # enforce orders
    df_all = df_all.copy()
    df_all["bin_label"] = pd.Categorical(df_all["bin_label"], categories=BIN_ORDER, ordered=True)
    models_sorted = sorted(df_all["model"].unique(), key=_model_order_key)
    df_all["model"] = pd.Categorical(df_all["model"], categories=models_sorted, ordered=True)

    # one figure per language pair
    for lp in sorted(df_all["langpair"].unique()):
        sub = df_all[df_all["langpair"] == lp].sort_values(["bin_label", "model"])
        plt.figure(figsize=(10, 5))
        ycol = f"{metric_name}_mean"
        if _HAVE_SEABORN:
            sns.barplot(data=sub, x="bin_label", y=ycol, hue="model")
        else:
            # Fallback: manual grouped bars
            bins = BIN_ORDER
            n_bins = len(bins)
            n_models = len(models_sorted)
            width = 0.8 / max(1, n_models)
            for mi, mname in enumerate(models_sorted):
                vals = [sub[(sub["bin_label"] == b) & (sub["model"] == mname)][ycol].mean() for b in bins]
                x = [i + mi * width for i in range(n_bins)]
                plt.bar(x, vals, width=width, label=mname)
            centers = [i + (n_models - 1) * width / 2 for i in range(n_bins)]
            plt.xticks(centers, bins)

        plt.title(f"{metric_name.upper()} mean by bin — {lp}")
        plt.xlabel("bin"); plt.ylabel(f"{metric_name.upper()} mean")
        plt.legend(title="model", ncol=2, fontsize=8)
        plt.tight_layout()
        fp = out_dir / f"{metric_name}_by_bin_{lp.replace('-', '_')}.png"
        plt.savefig(fp, dpi=200); plt.close()

    print(f"[cross-model] Wrote grouped {metric_name.upper()} plots to: {out_dir}")


def run_cross_model_plots(stat_dirs):
    """
    Collect & plot for the desired metrics.
    Change the list below if you want more/less metrics.
    """
    out_base = _common_base_dir(stat_dirs)
    for metric in ["comet", "chrf", "bleu"]:
        df = collect_metric_by_bin(stat_dirs, metric)
        plot_cross_model_metric(df, metric, out_base)


# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser(description="Plot in-bin statistics per model; plus cross-model COMET-by-bin plots.")
    ap.add_argument("inputs", nargs="+", help="Parent results folder(s) or in_bin_statistics folder(s).")
    args = ap.parse_args()

    stat_dirs = discover_stat_dirs(args.inputs)
    if not stat_dirs:
        print("No in_bin_statistics folders with CSVs found.")
        return

    if _HAVE_SEABORN:
        sns.set_context("talk")
        sns.set_style("whitegrid")

    # Per-model plots
    for stat_dir in stat_dirs:
        plot_one_model(stat_dir)

    # Cross-model COMET-by-bin plots (read summary.json from each model folder)
    # Cross-model plots from summary.json (COMET, CHRF, BLEU)
    run_cross_model_plots(stat_dirs)


if __name__ == "__main__":
    main()