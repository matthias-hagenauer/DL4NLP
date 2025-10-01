import json

REQUIRED_FIELDS = {"lp", "source", "target"}

def load_jsonl_pairs(path):
    """
    Load pairs from a JSONL file.
    Each line should be a JSON object with at least: lp, source, target.
    lp must be in the form "srcLang-tgtLang".
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
            # skip bad sources
            if rec["is_bad_source"] == True:
                continue
            # lp must split into two codes
            try:
                parts = str(rec["lp"]).lower().split("_")[0].split("-")
                src_code, tgt_code = parts[0], parts[1]
            except Exception:
                continue
            # meta = everything else
            meta = {}
            for k, v in rec.items():
                if k not in ("lp", "source", "target"):
                    meta[k] = v
            pairs.append({
                "src_lang": src_code,
                "tgt_lang": tgt_code,
                "src": str(rec["source"]),
                "tgt": str(rec["target"]),
                "meta": meta,
            })
    if not pairs:
        raise ValueError("No valid records found in %s" % path)
    return pairs

