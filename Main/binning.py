from __future__ import annotations
from typing import List, Tuple, Optional

Bin = Tuple[float, float]  

def parse_bins(spec: str) -> List[Bin]:
    bins: List[Bin] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            lo_s, hi_s = part.split("-")
            lo, hi = float(lo_s), float(hi_s)
        except Exception as e:
            raise ValueError(f"Invalid bin segment '{part}' in spec '{spec}'") from e
        if hi <= lo:
            raise ValueError(f"Invalid bin '{part}' (requires hi > lo)")
        bins.append((lo, hi))
    return bins


def assign_bin(value: Optional[float], bins: List[Bin]) -> str:
    if value is None:
        return "UNBINNED"
    for lo, hi in bins:
        if value >= lo and value <= hi:
            return f"({lo}-{hi}]"
    return "UNBINNED"


def coerce_esa(x: object) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)  
    except Exception:
        return None
