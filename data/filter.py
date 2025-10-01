#!/usr/bin/env python3
import json
import random
import argparse
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

# Targets you care about (exclude 'en')
KEEP_TARGETS = {"pt", "es", "fr", "de", "nl", "it", "ko", "zh", "ru"}

def parse_langs(langs_field: str) -> Tuple[str, str]:
    # Expected format "xx-yy"
    parts = langs_field.split("-")
    if len(parts) != 2:
        raise ValueError(f"Bad langs field: {langs_field}")
    return parts[0], parts[1]

def iter_jsonl(path: str) -> Iterable[Tuple[Dict, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield obj, line
            except Exception:
                # Skip malformed lines quietly
                continue

def filter_en_to_targets(input_path: str, targets: Set[str]) -> Iterable[Tuple[Dict, str]]:
    """
    Yield (obj, raw_line) for lines where src == 'en' and tgt in `targets`.
    """
    for obj, raw in iter_jsonl(input_path):
        try:
            s, t = parse_langs(obj["langs"])
            if s == "en" and t in targets:
                yield obj, raw
        except Exception:
            continue

def collect_by_tgt(filtered_iter: Iterable[Tuple[Dict, str]], targets: Set[str]):
    """
    Build mapping tgt_lang -> list of raw lines (dedup all_lines in arrival order).
    """
    by_tgt = defaultdict(list)
    all_lines, seen_raw = [], set()

    for obj, raw in filtered_iter:
        _, t = parse_langs(obj["langs"])
        if t in targets:
            by_tgt[t].append(raw)
        if raw not in seen_raw:
            seen_raw.add(raw)
            all_lines.append(raw)

    return by_tgt, all_lines

def write_lines(lines: List[str], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as out:
        for ln in lines:
            out.write(ln if ln.endswith("\n") else ln + "\n")

def balanced_sample(by_tgt: Dict[str, List[str]], targets: Set[str], n_per_lang: int, seed: int = None):
    """
    Greedy de-duplicated sampler: up to n_per_lang per TARGET language.
    A line appears at most once overall.
    """
    if seed is not None:
        random.seed(seed)

    # Ensure every target has a bucket
    buckets = {t: by_tgt.get(t, []).copy() for t in targets}
    for t in buckets:
        random.shuffle(buckets[t])

    chosen, chosen_set = [], set()
    counts = {t: 0 for t in targets}

    progress = True
    while progress:
        progress = False
        for t in targets:
            if counts[t] >= n_per_lang:
                continue
            bucket = buckets[t]
            while bucket and counts[t] < n_per_lang:
                ln = bucket.pop()
                if ln in chosen_set:
                    continue
                chosen.append(ln)
                chosen_set.add(ln)
                counts[t] += 1
                progress = True
                break

    return chosen, counts

def parse_lang_list(opt: str) -> Set[str]:
    if not opt:
        return set()
    return {x.strip() for x in opt.split(",") if x.strip()}

def main():
    ap = argparse.ArgumentParser(description="Filter en→X JSONL and (optionally) balance by TARGET language.")
    ap.add_argument("--input", required=True, help="Path to input .jsonl")
    ap.add_argument("--output", required=True, help="Path to output .jsonl")
    ap.add_argument("--mode", choices=["all", "balanced"], default="all",
                    help="all = write all en→target lines; balanced = up to n per TARGET")
    ap.add_argument("--n", type=int, default=100, help="Number of lines per TARGET language for balanced mode")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for balanced sampling")
    ap.add_argument("--targets", default="",
                    help="Comma-separated target langs to include (default: pt,es,fr,de,nl,it,ko,zh,ru)")
    args = ap.parse_args()

    targets = parse_lang_list(args.targets) or KEEP_TARGETS

    # 1) Filter to only en→target lines
    filtered_iter = filter_en_to_targets(args.input, targets)

    # 2) Build TARGET buckets + all filtered lines
    by_tgt, all_filtered_lines = collect_by_tgt(filtered_iter, targets)

    if args.mode == "all":
        write_lines(all_filtered_lines, args.output)
        print(f"Wrote {len(all_filtered_lines)} en→target lines to {args.output}")
    else:
        chosen, counts = balanced_sample(by_tgt, targets, args.n, seed=args.seed)
        write_lines(chosen, args.output)
        filled = sum(1 for t in targets if counts.get(t, 0) >= args.n)
        print(f"Wrote {len(chosen)} lines to {args.output}.")
        print(f"Targets fully filled: {filled}/{len(targets)}.")
        print(f"Per-target counts: {counts}")

if __name__ == "__main__":
    main()
