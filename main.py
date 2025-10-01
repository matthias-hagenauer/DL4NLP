import argparse
import csv
import json
import os

from data import load_jsonl_pairs
from model import build_model
from binning import parse_bins, assign_bin, coerce_esa
from eval import chrf_segment_scores, bleu_corpus, comet22_scores

# Extra import for new binning process
from binning import quantile_bin_tuples, balance_bins
import random

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def dump_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def dump_csv(path, rows):
    if not rows:
        return

    keys = set()
    for r in rows:
        for k in r.keys():
            keys.add(k)
    keys = sorted(keys)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            out = {}
            for k in keys:
                out[k] = r.get(k, "")
            w.writerow(out)

def valid_pred_ref(r):
    if not isinstance(r.get("pred"), str):
        return False
    if not isinstance(r.get("ref"), str):
        return False
    return r["ref"].strip() != ""

def mean_of(values):
    if not values:
        return None
    return sum(values) / float(len(values))

def group_mean(rows, metric_key, key_fn):
    groups = {}
    counts = {}

    for r in rows:
        g = key_fn(r)
        counts[g] = counts.get(g, 0) + 1

        v = None
        metrics = r.get("metrics")
        if isinstance(metrics, dict):
            v = metrics.get(metric_key, None)
        if isinstance(v, (int, float)):
            groups.setdefault(g, []).append(float(v))

    out = {}
    for g, c in counts.items():
        vals = groups.get(g, [])
        out[g] = {"count": c, "mean": mean_of(vals)}
    return out

def group_corpus_bleu(rows, key_fn):
    preds = {}
    refs = {}
    counts = {}

    for r in rows:
        g = key_fn(r)
        counts[g] = counts.get(g, 0) + 1

        if valid_pred_ref(r):
            preds.setdefault(g, []).append(r["pred"])
            refs.setdefault(g, []).append(r["ref"])

    out = {}
    for g in counts.keys():
        pv = preds.get(g, [])
        rv = refs.get(g, [])
        bleu_val = None
        if pv and rv and len(pv) == len(rv):
            bleu_val = float(bleu_corpus(pv, rv).get("bleu", 0.0))
        out[g] = {"count": counts[g], "bleu": bleu_val}
    return out

def main():
    ap = argparse.ArgumentParser(description="Translate, evaluate, and ESA-bin results (simplified).")

    ap.add_argument("--data", required=True, help="Path to JSONL with fields: src_lang, tgt_lang, src, tgt (+ meta.esa_score).")

    # Single switch: TM, TM_2bit, TM_3bit, TM_4bit, TM_5bit, TM_6bit, TM_8bit, or any HF id
    ap.add_argument("--model_id", default="TM",
                    help="Model id: TM (HF baseline), TM_2bit..TM_8bit (GGUF), or any HF model id.")

    ap.add_argument("--outdir", default=None, help="Output dir (default: results/<model_id>).")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--bins", choices=["interval", "interval_balanced", "quantile", "quantile_balanced"], default="interval",
                    help='interval: [0-33],(33-66],(66-100]; quantile: 3 equal-size bins (approximately); balanced: equal-size bins forced by random pruning to min size.')
    ap.add_argument("--eval_metrics", default="chrf", help="Comma-separated subset of {chrf,bleu,comet}.")
    ap.add_argument("--comet_gpus", type=int, default=1)
    ap.add_argument("--comet_batch", type=int, default=8)

    # GGUF optional overrides (only matter for TM_*bit)
    ap.add_argument("--gguf_repo", default=None, help="Override HF repo for GGUF.")
    ap.add_argument("--gguf_file", default=None, help="Override filename inside HF repo for GGUF.")
    ap.add_argument("--gguf_path", default=None, help="Override local path to .gguf.")
    ap.add_argument("--n_ctx", type=int, default=4096, help="llama.cpp context length")
    ap.add_argument("--n_gpu_layers", type=int, default=0, help="llama.cpp GPU offload layers (0=CPU).")

    args = ap.parse_args()

    # Default outdir if not provided
    if not args.outdir:
        args.outdir = f"results/{args.model_id}"
    ensure_dir(args.outdir)

    # 1) Load data
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

    ##### Binning Logic #####

    is_interval = args.bins.startswith("interval")
    is_balanced = args.bins.endswith("balanced")

    # Collect ESA scores from item meta (coerce to float, skip bad values)
    scores = []
    for it in items:
        meta = it.get("meta", {}) if isinstance(it.get("meta", {}), dict) else {}
        s = meta.get("esa_score")
        try:
            scores.append(float(s))
        except Exception:
            pass

    # Choose bin scheme
    if is_interval:
        # [0-33], (33-66], (66-100] as in the help text
        bin_tuples = parse_bins("0-33,33-66,66-100")
    else:
        # Quantile bins from data (q=3 ~ terciles)
        bin_tuples = quantile_bin_tuples(scores, q=3)

    # Assign difficulty label per item
    for it in items:
        meta = it.get("meta", {}) if isinstance(it.get("meta", {}), dict) else {}
        esa = coerce_esa(meta.get("esa_score"))
        it["difficulty"] = assign_bin(esa, bin_tuples)

    # Optional balancing (in-place; marks overflow items as UNBINNED)
    if is_balanced:
        balance_bins(bin_tuples, items, seed=42)

    #########################

    # 3) Attach ESA score/bin
    # bins = parse_bins(args.bins) no longer needed, see Binning Logic
    rows = []
    for idx, (it, pred) in enumerate(zip(items, preds)):
        meta = it.get("meta", {}) if isinstance(it.get("meta", {}), dict) else {}
        esa = coerce_esa(meta.get("esa_score"))
        label = it.get("difficulty")   # TODO needs a 'not == UNBINNED' check for balanced!!!!
        uid = meta.get("line_id", idx)
        src_lang = it.get("src_lang")
        tgt_lang = it.get("tgt_lang")
        langpair = f"{src_lang}-{tgt_lang}"

        rows.append({
            "id": uid,
            "langpair": langpair,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "src": it.get("src"),
            "ref": it.get("tgt"),
            "pred": pred,
            "esa_score": esa,
            "esa_bin": label,
            "meta": meta,
            "metrics": {},
            "binning": args.bins
        })

    # 4) Metrics
    metrics = []
    for m in args.eval_metrics.split(","):
        m = m.strip().lower()
        if m:
            metrics.append(m)

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

    # 5) Save per-row data
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
            "esa_score": r.get("esa_score"),
            "esa_bin": r.get("esa_bin"),
            "chrf": m.get("chrf"),
            "comet_seg": m.get("comet_seg"),
        })
    dump_csv(csv_path, rows_for_csv)

    # 6) Summaries TODO: Add std to summary
    def key_by_bin(r): return r.get("esa_bin", "UNBINNED")
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

    # 7) Write summary.json
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

    # 8) Console summary
    print("==== Info ===")
    print("Model:", args.model_id)
    print("Bins:", bin_tuples)
    print("Sample assignments:", [assign_bin(s, bin_tuples) for s in [0, 33, 66, 100]])
    print("Saved:", pred_path)
    print("Saved:", csv_path)
    print("Saved:", summary_path)
    print("\nESA bins:")

    bins_seen = {r.get("esa_bin") for r in rows}
    for b in sorted(bins_seen):
        n_in_bin = sum(1 for r in rows if r.get("esa_bin") == b)
        parts = [("{:>12}".format(str(b))) + "  n=" + str(n_in_bin)]
        if "chrf" in metrics_summary:
            parts.append("chrf=" + str(metrics_summary["chrf"]["by_bin"].get(b, {}).get("mean")))
        if "comet" in metrics_summary:
            parts.append("comet=" + str(metrics_summary["comet"]["by_bin"].get(b, {}).get("mean")))
        if "bleu" in metrics_summary:
            parts.append("bleu=" + str(metrics_summary["bleu"]["by_bin"].get(b, {}).get("bleu")))
        print("  " + "  ".join(parts))

if __name__ == "__main__":
    main()
