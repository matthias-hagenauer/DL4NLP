import argparse
import csv
import json
import os

from data import load_jsonl_pairs
from model import build_model
from eval import chrf_segment_scores, bleu_corpus, comet22_scores

from binning import parse_bins, assign_bin, coerce_esa, quantile_bin_tuples, balance_bins


# Utility functions
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def dump_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dump_csv(path, rows):
    if not rows:
        return

    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

# Pred utilituy functions
def valid_pred_ref(r):
    return isinstance(r.get("pred"), str) and isinstance(r.get("ref"), str) and r["ref"].strip() != ""


def mean_of(values):
    if not values:
        return None
    return sum(values) / float(len(values))


def group_mean(rows, metric_key, key_fn):
    groups, counts = {}, {}
    for r in rows:
        g = key_fn(r)
        counts[g] = counts.get(g, 0) + 1
        v = r.get("metrics", {}).get(metric_key)
        if isinstance(v, (int, float)):
            groups.setdefault(g, []).append(float(v))

    return {g: {"count": counts[g], "mean": mean_of(groups.get(g, []))} for g in counts}


def group_corpus_bleu(rows, key_fn):
    preds, refs, counts = {}, {}, {}
    for r in rows:
        g = key_fn(r)
        counts[g] = counts.get(g, 0) + 1
        if valid_pred_ref(r):
            preds.setdefault(g, []).append(r["pred"])
            refs.setdefault(g, []).append(r["ref"])
    out = {}
    for g in counts:
        pv, rv = preds.get(g, []), refs.get(g, [])
        bleu_val = float(bleu_corpus(pv, rv).get("bleu", 0.0)) if pv and rv and len(pv) == len(rv) else None
        out[g] = {"count": counts[g], "bleu": bleu_val}
    return out


# Binning utility functions
def collect_difficulty_scores(items):
    """
    Extract numeric difficulty scores from items. Accepts top-level 'difficulty_score'
    (preferred) and falls back to meta['difficulty_score'].
    """
    scores = []
    for it in items:
        s = it.get("difficulty_score")
        if s is None:
            meta = it.get("meta", {}) if isinstance(it.get("meta", {}), dict) else {}
            s = meta.get("difficulty_score")
        s = coerce_esa(s)  # generic float coercion
        if s is not None:
            scores.append(s)
    return scores


def choose_bins(items, bins_mode):
    """
    bins_mode in {"interval", "interval_balanced", "quantile", "quantile_balanced"}.
    For "interval*" we expect users to pass a parse-able spec if they override; we keep
    the legacy default of 3 bins with equal widths determined from the data if needed.
    """
    is_interval = bins_mode.startswith("interval")
    scores = collect_difficulty_scores(items)

    if is_interval:
        # If the user wants interval bins but didn't pass a spec elsewhere,
        # fall back to quantile as a sane default for arbitrary real-valued scores.
        # (Keeping interface simple: we use terciles.)
        return quantile_bin_tuples(scores, q=3)
    else:
        # Quantile terciles over observed scores (supports negatives, ties)
        return quantile_bin_tuples(scores, q=3)


def assign_difficulty_bins(items, bin_tuples, balanced: bool):
    """
    Mutates items:
      - sets item['difficulty'] = label produced by assign_bin(score, bin_tuples)
      - if balanced, calls balance_bins to mark overflow items as UNBINNED
    """
    # Initial assignment
    for it in items:
        s = it.get("difficulty_score")
        if s is None:
            meta = it.get("meta", {}) if isinstance(it.get("meta", {}), dict) else {}
            s = meta.get("difficulty_score")
        it["difficulty"] = assign_bin(coerce_esa(s), bin_tuples)

    if balanced:
        balance_bins(bin_tuples, items, seed=42)


# Main function
def main():
    ap = argparse.ArgumentParser(description="Translate, evaluate, and bin by difficulty.")
    ap.add_argument("--data", required=True, help="Path to NEW-schema JSONL (lp, source, target, ...).")
    # Model
    ap.add_argument("--model_id", default="TM",
                    help="TM (HF baseline), TM_2bit..TM_8bit (GGUF), or any HF model id.")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    # Outputs
    ap.add_argument("--outdir", default=None, help="Output dir (default: results/<model_id>).")
    # Binning
    ap.add_argument("--bins", choices=["interval", "interval_balanced", "quantile", "quantile_balanced"],
                    default="interval",
                    help="interval*: equal-sized groups via data-driven fallback; quantile*: terciles. *_balanced marks overflow items as UNBINNED.")
    # COMET config
    ap.add_argument("--eval_metrics", nargs="+", choices=["chrf", "bleu", "comet"], default=["chrf"],
                    help="Subset of metrics to compute.")
    ap.add_argument("--comet_gpus", type=int, default=1)
    ap.add_argument("--comet_batch", type=int, default=8)
    # GGUF (only if using TM_*bit)
    ap.add_argument("--gguf_repo", default=None)
    ap.add_argument("--gguf_file", default=None)
    ap.add_argument("--gguf_path", default=None)
    ap.add_argument("--n_ctx", type=int, default=4096)
    ap.add_argument("--n_gpu_layers", type=int, default=0)

    args = ap.parse_args()

    # Output dir
    if not args.outdir:
        args.outdir = f"results/{args.model_id}"
    ensure_dir(args.outdir)

    # 1) Load data (already normalized & filtered by data.py)
    items = load_jsonl_pairs(args.data)

    # 2) Build model & translate
    model = build_model(
        model_id=args.model_id,
        device_map=args.device_map,
        gguf_repo=args.gguf_repo,
        gguf_file=args.gguf_file,
        gguf_path=args.gguf_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
    )
    preds = model.translate_batch(items, max_new_tokens=args.max_new_tokens)

    # 3) Binning (clean)
    is_balanced = args.bins.endswith("balanced")
    bin_tuples = choose_bins(items, args.bins)
    assign_difficulty_bins(items, bin_tuples, balanced=is_balanced)

    # 4) Build per-row records
    rows = []
    for idx, (it, pred) in enumerate(zip(items, preds)):
        meta = it.get("meta", {}) if isinstance(it.get("meta", {}), dict) else {}
        # Stable ID: document_id:segment_id:lp  (falls back to idx if missing)
        doc_id = str(meta.get("document_id", ""))
        seg_id = str(meta.get("segment_id", ""))
        lp     = f"{it.get('src_lang','?')}-{it.get('tgt_lang','?')}"
        uid = f"{doc_id}:{seg_id}:{lp}" if (doc_id and seg_id) else str(idx)

        rows.append({
            "id": uid,
            "langpair": lp,
            "src_lang": it.get("src_lang"),
            "tgt_lang": it.get("tgt_lang"),
            "src": it.get("src"),
            "ref": it.get("tgt"),
            "pred": pred,
            "difficulty_score": it.get("difficulty_score", meta.get("difficulty_score")),
            "difficulty_bin": it.get("difficulty"),
            "meta": meta,
            "metrics": {},
            "binning": args.bins
        })

    # 5) Metrics
    metrics = [m.lower() for m in args.eval_metrics]
    if "chrf" in metrics:
        idxs = [i for i, r in enumerate(rows) if valid_pred_ref(r)]
        if idxs:
            preds_v = [rows[i]["pred"] for i in idxs]
            refs_v  = [rows[i]["ref"]  for i in idxs]
            seg_chrf = chrf_segment_scores(preds_v, refs_v)
            for k, score in zip(idxs, seg_chrf):
                rows[k]["metrics"]["chrf"] = score

    comet_overall = None
    if "comet" in metrics:
        idxs = [i for i, r in enumerate(rows) if valid_pred_ref(r) and isinstance(r.get("src"), str)]
        if idxs:
            srcs_v  = [rows[i]["src"]  for i in idxs]
            preds_v = [rows[i]["pred"] for i in idxs]
            refs_v  = [rows[i]["ref"]  for i in idxs]
            com = comet22_scores(srcs_v, preds_v, refs_v,
                                 gpus=args.comet_gpus, batch_size=args.comet_batch)
            comet_overall = float(com.get("system_score", 0.0))
            seg_scores = com.get("segment_scores", [])
            for k, score in zip(idxs, seg_scores):
                rows[k]["metrics"]["comet_seg"] = score

    overall_bleu = None
    if "bleu" in metrics:
        idxs = [i for i, r in enumerate(rows) if valid_pred_ref(r)]
        if idxs:
            preds_v = [rows[i]["pred"] for i in idxs]
            refs_v  = [rows[i]["ref"]  for i in idxs]
            overall_bleu = float(bleu_corpus(preds_v, refs_v).get("bleu", 0.0))

    # 6) Save per-row data
    pred_path = os.path.join(args.outdir, "predictions.jsonl")
    csv_path  = os.path.join(args.outdir, "rows.csv")
    dump_jsonl(pred_path, rows)

    rows_for_csv = []
    for r in rows:
        m = r.get("metrics", {})
        rows_for_csv.append({
            "id": r.get("id"),
            "langpair": r.get("langpair"),
            "src_lang": r.get("src_lang"),
            "tgt_lang": r.get("tgt_lang"),
            "difficulty_score": r.get("difficulty_score"),
            "difficulty_bin": r.get("difficulty_bin"),
            "chrf": m.get("chrf"),
            "comet_seg": m.get("comet_seg"),
        })
    dump_csv(csv_path, rows_for_csv)

    # 7) Summaries
    def key_by_bin(r): return r.get("difficulty_bin", "UNBINNED")
    def key_by_lp(r): return r.get("langpair", "unknown")
    def key_by_lp_bin(r): return key_by_lp(r) + " | " + key_by_bin(r)

    metrics_summary = {}
    if "chrf" in metrics:
        metrics_summary["chrf"] = {
            "overall": {
                "count": len(rows),
                "mean": mean_of([r["metrics"]["chrf"] for r in rows if isinstance(r.get("metrics", {}).get("chrf"), (int, float))])
            },
            "by_bin":          group_mean(rows, "chrf", key_by_bin),
            "by_langpair":     group_mean(rows, "chrf", key_by_lp),
            "by_langpair_bin": group_mean(rows, "chrf", key_by_lp_bin),
        }

    if "comet" in metrics:
        metrics_summary["comet"] = {
            "overall": {
                "count": len(rows),
                "mean": mean_of([r["metrics"]["comet_seg"] for r in rows if isinstance(r.get("metrics", {}).get("comet_seg"), (int, float))]),
                "system_score": comet_overall,
            },
            "by_bin":          group_mean(rows, "comet_seg", key_by_bin),
            "by_langpair":     group_mean(rows, "comet_seg", key_by_lp),
            "by_langpair_bin": group_mean(rows, "comet_seg", key_by_lp_bin),
        }

    if "bleu" in metrics:
        metrics_summary["bleu"] = {
            "overall": {"count": len(rows), "bleu": overall_bleu},
            "by_bin":          group_corpus_bleu(rows, key_by_bin),
            "by_langpair":     group_corpus_bleu(rows, key_by_lp),
            "by_langpair_bin": group_corpus_bleu(rows, key_by_lp_bin),
        }

    # 8) Write summary.json
    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "data": args.data,
                "model_id": args.model_id,
                "device_map": args.device_map,
                "max_new_tokens": args.max_new_tokens,
                "bins": args.bins,
                "eval_metrics": sorted(set(metrics)),
            },
            "counts": {"rows": len(rows)},
            "metrics": metrics_summary,
        }, f, ensure_ascii=False, indent=2)

    # 9) Console summary
    print("==== Info ===")
    print("Model:", args.model_id)
    print("Bins:", bin_tuples)
    # Show example assignments for three representative points (min/median/max if available)
    scores = collect_difficulty_scores(items)
    if scores:
        ex = sorted(scores)
        probe = [ex[0], ex[len(ex)//2], ex[-1]]
        print("Sample assignments:", [assign_bin(v, bin_tuples) for v in probe])
    print("Saved:", pred_path)
    print("Saved:", csv_path)
    print("Saved:", summary_path)
    print("\nDifficulty bins:")

    bins_seen = {r.get("difficulty_bin") for r in rows}
    for b in sorted(bins_seen):
        n_in_bin = sum(1 for r in rows if r.get("difficulty_bin") == b)
        parts = [("{:>18}".format(str(b))) + "  n=" + str(n_in_bin)]
        if "chrf" in metrics_summary:
            parts.append("chrf=" + str(metrics_summary["chrf"]["by_bin"].get(b, {}).get("mean")))
        if "comet" in metrics_summary:
            parts.append("comet=" + str(metrics_summary["comet"]["by_bin"].get(b, {}).get("mean")))
        if "bleu" in metrics_summary:
            parts.append("bleu=" + str(metrics_summary["bleu"]["by_bin"].get(b, {}).get("bleu")))
        print("  " + "  ".join(parts))


if __name__ == "__main__":
    main()