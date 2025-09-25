# main.py
# End-to-end: load → translate → (optional) evaluate → ESA-bin → save
# Summary includes, for each requested metric: overall, by ESA bin, by language pair, and their combination.

from __future__ import annotations
from typing import Any, Dict, List, Callable, Optional
import argparse
import csv
import json
import os
from statistics import mean
from collections import defaultdict

from data import load_jsonl_pairs
from models import build_model
from binning import parse_bins, assign_bin, coerce_esa
from eval import chrf_segment_scores, sentence_bleu_scores, bleu_corpus, comet22_scores


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dump_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _valid_pred_ref(r: Dict[str, Any]) -> bool:
    return isinstance(r.get("pred"), str) and isinstance(r.get("ref"), str) and r["ref"].strip() != ""


def _group_reduce_mean(
    rows: List[Dict[str, Any]],
    metric_key: str,
    key_fn: Callable[[Dict[str, Any]], str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Mean of a per-row metric (e.g., 'chrf', 'sbleu', 'comet_seg') grouped by key_fn.
    Returns {group: {"count": int, "mean": float|None}}
    """
    groups: Dict[str, List[float]] = defaultdict(list)
    counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        g = key_fn(r)
        counts[g] += 1
        v = r.get("metrics", {}).get(metric_key)
        if isinstance(v, (int, float)):
            groups[g].append(float(v))
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for g, c in counts.items():
        vals = groups.get(g, [])
        out[g] = {"count": c, "mean": (mean(vals) if vals else None)}
    return out


def _group_bleu(
    rows: List[Dict[str, Any]],
    key_fn: Callable[[Dict[str, Any]], str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Corpus BLEU per group using bleu_corpus on valid (pred, ref) pairs.
    Returns {group: {"count": int, "bleu": float|None}}
    """
    preds: Dict[str, List[str]] = defaultdict(list)
    refs: Dict[str, List[str]] = defaultdict(list)
    counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        g = key_fn(r)
        counts[g] += 1
        if _valid_pred_ref(r):
            preds[g].append(r["pred"])
            refs[g].append(r["ref"])
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for g in counts:
        pv, rv = preds.get(g, []), refs.get(g, [])
        bleu_val: Optional[float] = None
        if pv and rv and len(pv) == len(rv):
            bleu_val = float(bleu_corpus(pv, rv).get("bleu", 0.0))
        out[g] = {"count": counts[g], "bleu": bleu_val}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Translate, evaluate, and ESA-bin results.")
    ap.add_argument("--data", required=True, help="Path to JSONL with fields: langs, src, tgt (+ extras).")
    ap.add_argument("--outdir", required=True, help="Directory to write outputs.")
    ap.add_argument("--model_id", default="Unbabel/TowerInstruct-Mistral-7B-v0.2")
    ap.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"], help="bitsandbytes quantization mode.")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--bins", default="0-30,30-60,60-100", help='ESA bins like "0-30,30-60,60-100" (lo,hi] semantics).')
    ap.add_argument("--eval_metrics", default="chrf,sbleu", help="Subset of {chrf,sbleu,bleu,comet} (comma-separated).")
    ap.add_argument("--comet_gpus", type=int, default=1)
    ap.add_argument("--comet_batch", type=int, default=8)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # 1) Load data
    items = load_jsonl_pairs(args.data)

    # 2) Build model & translate
    model = build_model(model_id=args.model_id, quant=args.quant, device_map=args.device_map)
    preds = model.translate_batch(items, max_new_tokens=args.max_new_tokens)

    # 3) Attach ESA score/bin and prepare rows
    bins = parse_bins(args.bins)
    rows: List[Dict[str, Any]] = []
    for idx, (it, pred) in enumerate(zip(items, preds)):
        esa = coerce_esa(it.get("meta", {}).get("esa_score"))
        label = assign_bin(esa, bins)
        uid = it.get("meta", {}).get("line_id", idx)
        langpair = f"{it['src_lang']}→{it['tgt_lang']}"
        rows.append(
            {
                "id": uid,
                "langpair": langpair,
                "src_lang": it["src_lang"],
                "tgt_lang": it["tgt_lang"],
                "src": it["src"],
                "ref": it["tgt"],
                "pred": pred,
                "esa_score": esa,
                "esa_bin": label,
                "meta": it.get("meta", {}),
                "metrics": {},
            }
        )

    # 4) Metrics (optional, robust to missing items)
    want = {m.strip().lower() for m in args.eval_metrics.split(",") if m.strip()}

    # chrF (per segment)
    if "chrf" in want:
        idxs = [i for i, r in enumerate(rows) if _valid_pred_ref(r)]
        if idxs:
            preds_v = [rows[i]["pred"] for i in idxs]
            refs_v = [rows[i]["ref"] for i in idxs]
            seg_chrf = chrf_segment_scores(preds_v, refs_v)
            for i, s in zip(idxs, seg_chrf):
                rows[i]["metrics"]["chrf"] = s

    # sBLEU (per segment)
    if "sbleu" in want:
        idxs = [i for i, r in enumerate(rows) if _valid_pred_ref(r)]
        if idxs:
            preds_v = [rows[i]["pred"] for i in idxs]
            refs_v = [rows[i]["ref"] for i in idxs]
            seg_sbleu = sentence_bleu_scores(preds_v, refs_v)
            for i, s in zip(idxs, seg_sbleu):
                rows[i]["metrics"]["sbleu"] = s

    # COMET-22 (per segment + overall system score)
    comet_overall: Optional[float] = None
    if "comet" in want:
        idxs = [i for i, r in enumerate(rows) if _valid_pred_ref(r) and isinstance(r.get("src"), str)]
        if idxs:
            srcs_v = [rows[i]["src"] for i in idxs]
            preds_v = [rows[i]["pred"] for i in idxs]
            refs_v = [rows[i]["ref"] for i in idxs]
            com = comet22_scores(srcs_v, preds_v, refs_v, gpus=args.comet_gpus, batch_size=args.comet_batch)
            comet_overall = float(com.get("system_score", 0.0))
            for i, s in zip(idxs, com["segment_scores"]):
                rows[i]["metrics"]["comet_seg"] = s

    # BLEU (corpus-level)
    overall_bleu: Optional[float] = None
    if "bleu" in want:
        idxs = [i for i, r in enumerate(rows) if _valid_pred_ref(r)]
        if idxs:
            preds_v = [rows[i]["pred"] for i in idxs]
            refs_v = [rows[i]["ref"] for i in idxs]
            overall_bleu = float(bleu_corpus(preds_v, refs_v).get("bleu", 0.0))

    # 5) Persist per-row data (for later plotting)
    pred_path = os.path.join(args.outdir, "predictions.jsonl")
    csv_path = os.path.join(args.outdir, "rows.csv")  # was by_bin.csv; now a richer flat table
    dump_jsonl(pred_path, rows)
    dump_csv(
        csv_path,
        [
            {
                "id": r["id"],
                "langpair": r["langpair"],
                "src_lang": r["src_lang"],
                "tgt_lang": r["tgt_lang"],
                "esa_score": r["esa_score"],
                "esa_bin": r["esa_bin"],
                "chrf": r["metrics"].get("chrf"),
                "sbleu": r["metrics"].get("sbleu"),
                "comet_seg": r["metrics"].get("comet_seg"),
            }
            for r in rows
        ],
    )

    # 6) Build per-metric summaries
    # Key functions for groupings
    by_bin_key: Callable[[Dict[str, Any]], str] = lambda r: r.get("esa_bin", "UNBINNED")
    by_lp_key: Callable[[Dict[str, Any]], str] = lambda r: r.get("langpair", "unknown")
    by_lp_bin_key: Callable[[Dict[str, Any]], str] = lambda r: f"{r.get('langpair','unknown')} | {r.get('esa_bin','UNBINNED')}"

    metrics_summary: Dict[str, Any] = {}

    if "chrf" in want:
        metrics_summary["chrf"] = {
            "overall": {
                "count": len(rows),
                "mean": (mean([r["metrics"]["chrf"] for r in rows if r.get("metrics", {}).get("chrf") is not None])
                         if any(r.get("metrics", {}).get("chrf") is not None for r in rows) else None),
            },
            "by_bin": _group_reduce_mean(rows, "chrf", by_bin_key),
            "by_langpair": _group_reduce_mean(rows, "chrf", by_lp_key),
            "by_langpair_bin": _group_reduce_mean(rows, "chrf", by_lp_bin_key),
        }

    if "sbleu" in want:
        metrics_summary["sbleu"] = {
            "overall": {
                "count": len(rows),
                "mean": (mean([r["metrics"]["sbleu"] for r in rows if r.get("metrics", {}).get("sbleu") is not None])
                         if any(r.get("metrics", {}).get("sbleu") is not None for r in rows) else None),
            },
            "by_bin": _group_reduce_mean(rows, "sbleu", by_bin_key),
            "by_langpair": _group_reduce_mean(rows, "sbleu", by_lp_key),
            "by_langpair_bin": _group_reduce_mean(rows, "sbleu", by_lp_bin_key),
        }

    if "comet" in want:
        metrics_summary["comet"] = {
            "overall": {
                "count": len(rows),
                "mean": (mean([r["metrics"]["comet_seg"] for r in rows if r.get("metrics", {}).get("comet_seg") is not None])
                         if any(r.get("metrics", {}).get("comet_seg") is not None for r in rows) else None),
                "system_score": comet_overall,
            },
            "by_bin": _group_reduce_mean(rows, "comet_seg", by_bin_key),
            "by_langpair": _group_reduce_mean(rows, "comet_seg", by_lp_key),
            "by_langpair_bin": _group_reduce_mean(rows, "comet_seg", by_lp_bin_key),
        }

    if "bleu" in want:
        metrics_summary["bleu"] = {
            "overall": {"count": len(rows), "bleu": overall_bleu},
            "by_bin": _group_bleu(rows, by_bin_key),
            "by_langpair": _group_bleu(rows, by_lp_key),
            "by_langpair_bin": _group_bleu(rows, by_lp_bin_key),
        }

    # 7) Write summary.json with config + metrics
    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "data": args.data,
                    "model_id": args.model_id,
                    "quant": args.quant,
                    "device_map": args.device_map,
                    "max_new_tokens": args.max_new_tokens,
                    "bins": args.bins,
                    "eval_metrics": sorted(list(want)),
                },
                "counts": {"rows": len(rows)},
                "metrics": metrics_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 8) Console summary (brief)
    print("Saved:", pred_path)
    print("Saved:", csv_path)
    print("Saved:", summary_path)
    print("\nESA bins (avg chrF / sBLEU / COMET / BLEU if requested):")
    bins_sorted = sorted(set(r["esa_bin"] for r in rows))
    for b in bins_sorted:
        parts = [f"{b:>12}  n={len([r for r in rows if r['esa_bin']==b])}"]
        if "chrf" in metrics_summary:
            parts.append(f"chrf={metrics_summary['chrf']['by_bin'].get(b, {}).get('mean')}")
        if "sbleu" in metrics_summary:
            parts.append(f"sbleu={metrics_summary['sbleu']['by_bin'].get(b, {}).get('mean')}")
        if "comet" in metrics_summary:
            parts.append(f"comet={metrics_summary['comet']['by_bin'].get(b, {}).get('mean')}")
        if "bleu" in metrics_summary:
            parts.append(f"bleu={metrics_summary['bleu']['by_bin'].get(b, {}).get('bleu')}")
        print("  " + "  ".join(parts))


if __name__ == "__main__":
    main()