import json

REQUIRED_FIELDS = {"lp", "source", "target"}

def _parse_lp(lp_field: str):
    """
    Parse language pair in new schema ("xx-yy" or "xx-yy_REG") and
    return (src_lang, tgt_lang_base), both lowercased.

    Examples:
      - "en-zh"     -> ("en", "zh")
      - "en-zh_CN"  -> ("en", "zh")
      - "en-nl_NL"  -> ("en", "nl")
      - "EN-DE_DE"  -> ("en", "de")
    """
    lp = str(lp_field).strip().lower()
    parts = lp.split("-")
    if len(parts) != 2:
        raise ValueError(f"Bad lp field: {lp_field}")
    src = parts[0]
    tgt_full = parts[1]
    tgt_base = tgt_full.split("_", 1)[0]  # strip regional suffix if present
    return src, tgt_base

def load_jsonl_pairs(path):
    """
    Load pairs from a JSONL file.
    Each line must be a JSON object with at least: lp, source, target.
    - `lp` like "en-zh" or "en-zh_CN" (we normalize target to base: "zh")
    - `source` (string) becomes 'src'
    - `target` (string) becomes 'tgt'

    Returns a list of dicts with:
      {
        "src_lang": str,       # e.g., "en"
        "tgt_lang": str,       # e.g., "zh" (region stripped)
        "src": str,            # from "source"
        "tgt": str,            # from "target"
        "difficulty_score": float | None,  # promoted for binning convenience
        "meta": { ... }        # everything else, including document_id, segment_id, domain, etc.
      }

    Notes:
    - Rows with is_bad_source == True are skipped.
    - Non-string source/target are coerced to str.
    - We do NOT deduplicate here; keep (document_id, segment_id, lp) in meta for stable IDs upstream.
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

            # Must contain required fields
            if not all(k in rec for k in REQUIRED_FIELDS):
                continue

            # Skip known-bad sources if flagged
            if isinstance(rec.get("is_bad_source"), bool) and rec["is_bad_source"]:
                continue

            # Parse language pair and normalize target (strip regional suffix)
            try:
                src_code, tgt_code = _parse_lp(rec["lp"])
            except Exception:
                continue

            # Promote difficulty_score to top-level float (if present/parsable)
            diff_score = rec.get("difficulty_score", None)
            if diff_score is not None:
                try:
                    diff_score = float(diff_score)
                except Exception:
                    diff_score = None

            # Build meta = everything except the three core fields (lp, source, target)
            meta = {}
            for k, v in rec.items():
                if k in ("lp", "source", "target"):
                    continue
                meta[k] = v

            pairs.append({
                "src_lang": src_code,
                "tgt_lang": tgt_code,
                "src": str(rec.get("source", "")),
                "tgt": str(rec.get("target", "")),
                "difficulty_score": diff_score,
                "meta": meta,
            })

    if not pairs:
        raise ValueError(f"No valid records found in {path}")

    return pairs