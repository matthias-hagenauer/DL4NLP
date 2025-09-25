import json

REQUIRED_FIELDS = {"langs", "src", "tgt"}

def load_jsonl_pairs(path):
    """
    Load pairs from a JSONL file.
    Each line should be a JSON object with at least: langs, src, tgt.
    langs must be in the form "srcLang-tgtLang".
    Returns a list of dicts with: src_lang, tgt_lang, src, tgt, meta.
    """
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            # must contain required fields
            if not all(k in rec for k in REQUIRED_FIELDS):
                continue

            # langs must split into two codes
            try:
                parts = str(rec["langs"]).lower().split("-")
                src_code, tgt_code = parts[0], parts[1]
            except Exception:
                continue

            # meta = everything else
            meta = {}
            for k, v in rec.items():
                if k not in ("langs", "src", "tgt"):
                    meta[k] = v

            pairs.append({
                "src_lang": src_code,
                "tgt_lang": tgt_code,
                "src": str(rec["src"]),
                "tgt": str(rec["tgt"]),
                "meta": meta,
            })

    if not pairs:
        raise ValueError("No valid records found in %s" % path)

    return pairs
