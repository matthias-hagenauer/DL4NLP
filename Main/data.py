from __future__ import annotations
from typing import Any, Dict, List, TypedDict
import json

class Pair(TypedDict):
    src_lang: str
    tgt_lang: str
    src: str
    tgt: str
    meta: Dict[str, Any]


REQUIRED_FIELDS = {"langs", "src", "tgt"}


def load_jsonl_pairs(path: str) -> List[Pair]:
    pairs: List[Pair] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            # Basic schema check
            if not REQUIRED_FIELDS.issubset(rec):
                continue

            # Parse "en-ru" → ("en","ru")
            try:
                src_code, tgt_code = str(rec["langs"]).lower().split("-")
            except Exception:
                continue

            # Preserve any extra fields as metadata
            meta: Dict[str, Any] = {k: v for k, v in rec.items() if k not in {"langs", "src", "tgt"}}

            pairs.append(
                Pair(
                    src_lang=src_code,
                    tgt_lang=tgt_code,
                    src=str(rec["src"]),
                    tgt=str(rec["tgt"]),
                    meta=meta,
                )
            )

    if not pairs:
        raise ValueError(f"No valid records found in {path}")

    return pairs