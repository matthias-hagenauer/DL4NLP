#!/usr/bin/env python3
import json
import random
import argparse
from collections import defaultdict

KEEP_LANGS = {"en", "pt", "es", "fr", "de", "nl", "it", "ko", "zh", "ru"}

def parse_langs(langs_field: str):
    # Expected format "xx-yy"
    parts = langs_field.split("-")
    if len(parts) != 2:
        raise ValueError(f"Bad langs field: {langs_field}")
    return parts[0], parts[1]

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield obj, line
            except Exception as e:
                # Skip malformed lines quietly; you can print if you want
                # print(f"Skipping line {i}: {e}")
                continue

def filter_both_in_set(input_path):
    """Yield (obj, raw_line) where both langs are in KEEP_LANGS."""
    for obj, raw in iter_jsonl(input_path):
        try:
            s, t = parse_langs(obj["langs"])
            if s in KEEP_LANGS and t in KEEP_LANGS:
                yield obj, raw
        except Exception:
            continue

def collect_by_language(filtered_iter):
    """
    Build a mapping lang -> list of raw lines that contain that lang (src or tgt).
    Also return all filtered raw lines (deduped) in arrival order.
    """
    by_lang = defaultdict(list)
    all_lines = []
    seen_raw = set()

    for obj, raw in filtered_iter:
        s, t = parse_langs(obj["langs"])
        # Add to lang buckets if the language is one we keep
        if s in KEEP_LANGS:
            by_lang[s].append(raw)
        if t in KEEP_LANGS:
            by_lang[t].append(raw)
        # Track all filtered lines (dedup)
        if raw not in seen_raw:
            seen_raw.add(raw)
            all_lines.append(raw)

    return by_lang, all_lines

def write_lines(lines, output_path):
    with open(output_path, "w", encoding="utf-8") as out:
        for ln in lines:
            out.write(ln if ln.endswith("\n") else ln + "\n")

def balanced_sample(by_lang, n_per_lang, seed=None):
    """
    Greedy de-duplicated sampler: per language, pick lines containing that language
    until hitting n_per_lang or exhausting. A line is included at most once overall.
    """
    if seed is not None:
        random.seed(seed)

    # Shuffle each language bucket for randomness
    buckets = {lang: by_lang.get(lang, []).copy() for lang in KEEP_LANGS}
    for lang in buckets:
        random.shuffle(buckets[lang])

    chosen = []
    chosen_set = set()
    counts = {lang: 0 for lang in KEEP_LANGS}

    # Pass per language, fill greedily without duplicates
    # Keep looping while any language still needs lines and has candidates
    progress = True
    while progress:
        progress = False
        for lang in KEEP_LANGS:
            if counts[lang] >= n_per_lang:
                continue
            bucket = buckets[lang]
            # Pop lines until we find one not yet chosen
            while bucket and counts[lang] < n_per_lang:
                ln = bucket.pop()
                if ln in chosen_set:
                    continue
                chosen.append(ln)
                chosen_set.add(ln)
                counts[lang] += 1
                progress = True
                break  # move to next language

    return chosen, counts

def main():
    ap = argparse.ArgumentParser(description="Filter and (optionally) balance a JSONL of bilingual lines.")
    ap.add_argument("--input", required=True, help="Path to input .jsonl")
    ap.add_argument("--output", required=True, help="Path to output .jsonl")
    ap.add_argument("--mode", choices=["all", "balanced"], default="all",
                    help="all = write all filtered lines (both langs in set); balanced = n per language")
    ap.add_argument("--n", type=int, default=100, help="Number of lines per language for balanced mode")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for balanced sampling")
    args = ap.parse_args()

    # 1) Filter to only pairs where BOTH languages are in KEEP_LANGS
    filtered_iter = filter_both_in_set(args.input)

    # 2) Build lang buckets (for balanced) and collect all filtered lines
    by_lang, all_filtered_lines = collect_by_language(filtered_iter)

    if args.mode == "all":
        # Write all filtered lines (like earlier)
        write_lines(all_filtered_lines, args.output)
        print(f"Wrote {len(all_filtered_lines)} filtered lines to {args.output}")
    else:
        # Balanced mode
        chosen, counts = balanced_sample(by_lang, args.n, seed=args.seed)
        write_lines(chosen, args.output)
        filled = sum(1 for v in counts.values() if v >= args.n)
        print(f"Wrote {len(chosen)} lines to {args.output}. "
              f"Languages fully filled: {filled}/10. Per-lang counts: {counts}")

if __name__ == "__main__":
    main()
