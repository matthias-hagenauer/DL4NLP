#!/usr/bin/env python3
import json
import argparse

def main():
    ap = argparse.ArgumentParser(description="Min–max normalize difficulty_score to [0,1] and overwrite the field.")
    ap.add_argument("--input", required=True, help="Path to input JSONL (new schema).")
    ap.add_argument("--output", required=True, help="Path to output JSONL (normalized copy).")
    args = ap.parse_args()

    # Read all and collect scores
    records = []
    scores = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            records.append(obj)
            scores.append(float(obj["difficulty_score"]))

    mn = min(scores)
    mx = max(scores)
    rng = mx - mn

    # Normalize in-place (overwrite difficulty_score)
    with open(args.output, "w", encoding="utf-8") as out:
        if rng == 0.0:
            for obj in records:
                obj["difficulty_score"] = 0.0
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        else:
            for obj in records:
                s = float(obj["difficulty_score"])
                obj["difficulty_score"] = (s - mn) / rng
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote: {args.output}")

if __name__ == "__main__":
    main()
