#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple, Iterable
from collections import defaultdict, Counter

# --------------------- small utilities ---------------------

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                s2 = s.strip('"')
                try:
                    yield json.loads(s2)
                except Exception:
                    continue

def safefloat(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def tokenize_simple(text: str) -> List[str]:
    return (text or "").strip().split()

def char_len(text: str) -> int:
    return len(text or "")

def punct_density(text: str) -> float:
    if not text:
        return 0.0
    punct = sum(ch in ".,;:!?\"'()[]{}–—-…" for ch in text)
    return punct / max(1, len(text))

def sentence_count(text: str) -> int:
    if not text:
        return 0
    return sum(ch in ".?!" for ch in text)

def mean_std(vals: List[float]) -> Tuple[float, float]:
    v = [x for x in vals if isinstance(x, (int, float)) and not math.isnan(x)]
    if not v:
        return float("nan"), float("nan")
    m = sum(v) / len(v)
    var = sum((x - m) ** 2 for x in v) / len(v)
    return m, math.sqrt(var)

def median(vals: List[float]) -> float:
    v = sorted(x for x in vals if isinstance(x, (int, float)) and not math.isnan(x))
    if not v:
        return float("nan")
    n = len(v)
    mid = n // 2
    if n % 2 == 1:
        return v[mid]
    return 0.5 * (v[mid - 1] + v[mid])

def pearson(x: List[float], y: List[float]) -> float:
    paired = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    if not paired:
        return float("nan")
    xs, ys = zip(*paired)
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    denx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    deny = math.sqrt(sum((b - my) ** 2 for b in ys))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)

def print_table(rows: List[Dict[str, Any]], key_cols: List[str], value_cols: List[str], title: str):
    if not rows:
        return
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))
    cols = key_cols + value_cols
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = " | ".join(f"{c:{widths[c]}}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = " | ".join(f"{str(r.get(c, '')):{widths[c]}}" for c in cols)
        print(line)

# --------------------- per-bin logic ---------------------

def summarize_bin(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    diff = [safefloat(r.get("difficulty_score")) for r in records]

    srcs  = [r.get("src",  "") for r in records]
    refs  = [r.get("ref",  "") for r in records]
    preds = [r.get("pred", "") for r in records]

    src_tok  = [len(tokenize_simple(s)) for s in srcs]
    ref_tok  = [len(tokenize_simple(s)) for s in refs]
    pred_tok = [len(tokenize_simple(s)) for s in preds]

    src_chr  = [char_len(s) for s in srcs]
    ref_chr  = [char_len(s) for s in refs]
    pred_chr = [char_len(s) for s in preds]

    src_punct  = [punct_density(s) for s in srcs]
    ref_punct  = [punct_density(s) for s in refs]
    pred_punct = [punct_density(s) for s in preds]

    src_sents  = [sentence_count(s) for s in srcs]
    ref_sents  = [sentence_count(s) for s in refs]
    pred_sents = [sentence_count(s) for s in preds]

    exact_match = [1.0 if (r.get("pred") or "") == (r.get("ref") or "") else 0.0 for r in records]

    # dynamic metrics
    metric_keys = set()
    for r in records:
        metric_keys.update((r.get("metrics") or {}).keys())
    metric_vals: Dict[str, List[float]] = {k: [] for k in sorted(metric_keys)}
    for r in records:
        m = r.get("metrics") or {}
        for k in metric_keys:
            metric_vals[k].append(safefloat(m.get(k)))

    out: Dict[str, Any] = {
        "count": len(records),
        "difficulty_score_mean": mean_std(diff)[0],
        "difficulty_score_median": median(diff),
        "difficulty_score_std": mean_std(diff)[1],

        "src_tokens_mean": mean_std(src_tok)[0],
        "src_tokens_std":  mean_std(src_tok)[1],
        "ref_tokens_mean": mean_std(ref_tok)[0],
        "ref_tokens_std":  mean_std(ref_tok)[1],
        "pred_tokens_mean": mean_std(pred_tok)[0],
        "pred_tokens_std":  mean_std(pred_tok)[1],

        "src_chars_mean": mean_std(src_chr)[0],
        "src_chars_std":  mean_std(src_chr)[1],
        "ref_chars_mean": mean_std(ref_chr)[0],
        "ref_chars_std":  mean_std(ref_chr)[1],
        "pred_chars_mean": mean_std(pred_chr)[0],
        "pred_chars_std":  mean_std(pred_chr)[1],

        "src_punct_density_mean": mean_std(src_punct)[0],
        "ref_punct_density_mean": mean_std(ref_punct)[0],
        "pred_punct_density_mean": mean_std(pred_punct)[0],

        "src_sentence_count_mean":  mean_std(src_sents)[0],
        "ref_sentence_count_mean":  mean_std(ref_sents)[0],
        "pred_sentence_count_mean": mean_std(pred_sents)[0],

        "exact_match_rate": mean_std(exact_match)[0],
    }

    for k, vals in metric_vals.items():
        m, s = mean_std(vals)
        out[f"{k}_mean"] = m
        out[f"{k}_std"]  = s

    out["corr_diff__src_tokens"]  = pearson(diff, [float(x) for x in src_tok])
    out["corr_diff__ref_tokens"]  = pearson(diff, [float(x) for x in ref_tok])
    out["corr_diff__pred_tokens"] = pearson(diff, [float(x) for x in pred_tok])
    for k, vals in metric_vals.items():
        out[f"corr_diff__{k}"] = pearson(diff, vals)

    return out

def langpair_counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter()
    for r in records:
        lp = r.get("langpair") or f"{r.get('src_lang','?')}-{r.get('tgt_lang','?')}"
        c[lp] += 1
    return dict(c)

# --------------------- I/O helpers ---------------------

def find_jsonl(paths: List[Path]) -> List[Path]:
    found: List[Path] = []
    for p in paths:
        if p.is_file() and p.suffix.lower() in [".jsonl", ".json"]:
            found.append(p)
        elif p.is_dir():
            found.extend(list(p.rglob("predictions.jsonl")))
    # de-dup + sorted
    return sorted({str(p): p for p in found}.values())

def write_csv(rows: List[Dict[str, Any]], outpath: Path):
    if not rows:
        return
    cols = sorted({k for r in rows for k in r.keys()})
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

# --------------------- per-file (per-model) run ---------------------

def analyze_one_file(preds_path: Path):
    model_dir = preds_path.parent
    model_name = model_dir.name  # e.g., TM_2bit

    # collect by bin for this file only
    by_bin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in iter_jsonl(preds_path):
        b = rec.get("difficulty_bin") or "UNKNOWN"
        by_bin[b].append(rec)

    # stable bin order by lower bound if expressed like "(a-b]"
    def bin_key(b: str) -> float:
        try:
            core = b.strip("()[]")
            lo = core.split("-")[0]
            return float(lo)
        except Exception:
            return float("nan")

    summary_rows: List[Dict[str, Any]] = []
    corr_rows: List[Dict[str, Any]] = []
    lp_rows: List[Dict[str, Any]] = []

    for b in sorted(by_bin.keys(), key=bin_key):
        records = by_bin[b]
        summ = summarize_bin(records)
        summ["model"] = model_name
        summ["difficulty_bin"] = b
        summary_rows.append(summ)

        corr = {k: v for k, v in summ.items() if k.startswith("corr_diff__")}
        corr["model"] = model_name
        corr["difficulty_bin"] = b
        corr_rows.append(corr)

        for lp, cnt in sorted(langpair_counts(records).items()):
            lp_rows.append({"model": model_name, "difficulty_bin": b, "langpair": lp, "count": cnt})

    # write three CSVs into this model folder
    stats_dir = model_dir / "in_bin_statistics"
    write_csv(summary_rows, stats_dir / "bin_summary.csv")
    write_csv(corr_rows,    stats_dir / "metric_correlations.csv")
    write_csv(lp_rows,      stats_dir / "langpair_breakdown.csv")

    # console preview
    key_cols = ["model", "difficulty_bin", "count"]
    val_cols = [
        "difficulty_score_mean", "difficulty_score_std",
        "src_tokens_mean", "ref_tokens_mean", "pred_tokens_mean",
        "src_chars_mean", "ref_chars_mean", "pred_chars_mean",
        "exact_match_rate",
    ]
    print_table(summary_rows, key_cols, val_cols, title=f"Per-bin summary (preview) — {model_name}")

    metric_cols = sorted({k for r in summary_rows for k in r.keys()
                          if k.endswith("_mean") and not k.startswith(("src_", "ref_", "pred_", "difficulty_"))})
    if metric_cols:
        print_table(summary_rows, key_cols, [c for c in metric_cols if c.endswith("_mean")],
                    title=f"Per-bin metric means (preview) — {model_name}")

    print(f"\nWrote for {model_name}:")
    print(f" - {model_dir / 'bin_summary.csv'}")
    print(f" - {model_dir / 'metric_correlations.csv'}")
    print(f" - {model_dir / 'langpair_breakdown.csv'}\n")

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Analyze predictions per difficulty_bin and write CSVs next to each predictions.jsonl.")
    ap.add_argument("inputs", nargs="+", help="Paths to predictions.jsonl OR directories like results/TM_2bit/")
    args = ap.parse_args()

    jsonl_paths = find_jsonl([Path(x) for x in args.inputs])
    if not jsonl_paths:
        print("No predictions.jsonl files found.")
        return

    for p in jsonl_paths:
        analyze_one_file(p)

if __name__ == "__main__":
    main()