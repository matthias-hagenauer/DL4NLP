import json
import random
import argparse
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

KEEP_TARGETS = {"de", "es", "zh", "nl"}

def parse_lp(lp_field: str) -> Tuple[str, str, str]:
    """
    Parse language pair in new schema ("xx-yy" or "xx-yy_REG") and
    return (src, tgt_base, normalized_pair).
    Examples:
      - "en-zh"     -> ("en", "zh", "en-zh")
      - "en-zh_CN"  -> ("en", "zh", "en-zh")
      - "en-de_DE"  -> ("en", "de", "en-de")
    """
    parts = lp_field.split("-")
    if len(parts) != 2:
        raise ValueError(f"Bad lp field: {lp_field}")
    src = parts[0]
    tgt_full = parts[1]
    tgt_base = tgt_full.split("_", 1)[0]  # strip regional suffix if present
    return src, tgt_base, f"{src}-{tgt_base}"

def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def filter_en_to_targets(input_path: str, targets: Set[str]) -> Iterable[Dict]:
    """
    Yield JSON objects where src == 'en' and tgt_base in `targets`.
    Normalize 'lp' before yielding.
    """
    for obj in iter_jsonl(input_path):
        try:
            src, tgt_base, norm_pair = parse_lp(obj["lp"])
            if src == "en" and tgt_base in targets:
                obj["lp"] = norm_pair  
                yield obj
        except Exception:
            continue

def collect_by_tgt(filtered_iter: Iterable[Dict], targets: Set[str]):
    """
    Build mapping tgt_lang -> list of JSON objs; also return all objs in arrival order.
    Deduplicate by (document_id, segment_id, lp) so different targets for the same segment are kept.
    """
    by_tgt: Dict[str, List[Dict]] = defaultdict(list)
    all_objs: List[Dict] = []
    seen_keys = set()

    for obj in filtered_iter:
        _, tgt_base, _ = parse_lp(obj["lp"])
        key = (obj.get("document_id"), obj.get("segment_id"), obj["lp"])
        if key in seen_keys:
            continue
        seen_keys.add(key)

        if tgt_base in targets:
            by_tgt[tgt_base].append(obj)
        all_objs.append(obj)

    return by_tgt, all_objs

def write_objs(objs: List[Dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as out:
        for obj in objs:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

def balanced_sample(by_tgt: Dict[str, List[Dict]], targets: Set[str], n_per_lang: int, seed: int = None):
    """
    Greedy de-duplicated sampler: up to n_per_lang per TARGET language.
    A JSON object appears at most once overall (by doc_id, segment_id, lp).
    """
    if seed is not None:
        random.seed(seed)

    buckets = {t: by_tgt.get(t, []).copy() for t in targets}
    for t in buckets:
        random.shuffle(buckets[t])

    chosen: List[Dict] = []
    seen_keys = set()
    counts = {t: 0 for t in targets}

    progress = True
    while progress:
        progress = False
        for t in targets:
            if counts[t] >= n_per_lang:
                continue
            bucket = buckets[t]
            while bucket and counts[t] < n_per_lang:
                obj = bucket.pop()
                key = (obj.get("document_id"), obj.get("segment_id"), obj["lp"])
                if key in seen_keys:
                    continue
                chosen.append(obj)
                seen_keys.add(key)
                counts[t] += 1
                progress = True
                break

    return chosen, counts

def parse_lang_list(opt: str) -> Set[str]:
    if not opt:
        return set()
    return {x.strip() for x in opt.split(",") if x.strip()}

def main():
    ap = argparse.ArgumentParser(
        description="Filter EN→{DE,ES,ZH,NL} from NEW JSONL schema, normalize lp, and (optionally) balance by target language."
    )
    ap.add_argument("--input", required=False, default="data/wmt24_estimated.jsonl",
                    help="Path to input .jsonl (default: data/wmt24_estimated.jsonl)")
    ap.add_argument("--output", required=True, help="Path to output .jsonl")
    ap.add_argument("--mode", choices=["all", "balanced"], default="all",
                    help="all = write all EN→target lines; balanced = up to n per TARGET")
    ap.add_argument("--n", type=int, default=100,
                    help="Number of lines per TARGET language for balanced mode")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for balanced sampling")
    ap.add_argument("--targets", default="",
                    help="Comma-separated target langs to include (base codes only). Default: de,es,zh,nl")
    args = ap.parse_args()

    targets = parse_lang_list(args.targets) or KEEP_TARGETS

    # 1. Filter to only EN→target lines (normalize 'lp')
    filtered_iter = filter_en_to_targets(args.input, targets)

    # 2. Build TARGET buckets + all filtered objs (dedup per doc+seg+lp)
    by_tgt, all_filtered_objs = collect_by_tgt(filtered_iter, targets)

    if args.mode == "all":
        write_objs(all_filtered_objs, args.output)
        print(f"Wrote {len(all_filtered_objs)} en→target lines to {args.output}")
    else:
        chosen, counts = balanced_sample(by_tgt, targets, args.n, seed=args.seed)
        write_objs(chosen, args.output)
        filled = sum(1 for t in targets if counts.get(t, 0) >= args.n)
        print(f"Wrote {len(chosen)} lines to {args.output}.")
        print(f"Targets fully filled: {filled}/{len(targets)}.")
        print(f"Per-target counts: {counts}")

if __name__ == "__main__":
    main()
