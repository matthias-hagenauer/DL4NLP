# binning.py
# Typed helpers for ESA-based binning.

from __future__ import annotations
from typing import List, Tuple, Optional


Bin = Tuple[float, float]  # (lo, hi] semantics


def parse_bins(spec: str) -> List[Bin]:
    """
    Parse a spec like "0-30,30-60,60-100" into a list of (lo, hi] intervals.

    Rules:
      - Whitespace is ignored around parts.
      - Each part must be "lo-hi" with hi > lo.
      - Intervals are interpreted as (lo, hi] (open on lo, closed on hi).

    Raises:
      ValueError if a segment is malformed or hi <= lo.
    """
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
    """
    Assign a numeric value to the first (lo, hi] bin that contains it.
    Returns "UNBINNED" if value is None or falls outside all bins.
    """
    if value is None:
        return "UNBINNED"
    for lo, hi in bins:
        if value > lo and value <= hi:
            return f"({lo}-{hi}]"
    return "UNBINNED"


def coerce_esa(x: object) -> Optional[float]:
    """
    Convert an ESA score (possibly a string) to float, or None if missing/invalid.
    Examples:
      coerce_esa("59") -> 59.0
      coerce_esa(42)   -> 42.0
      coerce_esa(None) -> None
      coerce_esa("n/a")-> None
    """
    if x is None:
        return None
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return None
