import math
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

def parse_bins(spec: str) -> List[Tuple[float, float]]:
    """
    Parse a string into a list of (lo, hi) float tuples.

    Notes:
      - Bins are numeric intervals; any real values are allowed (negatives too).
      - Each segment must be 'lo-hi' with hi > lo.
    """
    bins: List[Tuple[float, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            lo_s, hi_s = part.split("-")
            lo, hi = float(lo_s), float(hi_s)
        except Exception:
            raise ValueError(f"Invalid bin segment '{part}' in spec '{spec}'")
        if hi <= lo:
            raise ValueError(f"Invalid bin '{part}' (requires hi > lo)")
        bins.append((lo, hi))
    return bins


def assign_bin(value: Optional[float], bins: Sequence[Tuple[float, float]]) -> str:
    """
    Assign a numeric value to a bin using (lo, hi] semantics:
      - First bin is [lo, hi]
      - Subsequent bins are (lo, hi]
    Returns a label like "(lo-hi]" or "UNBINNED".
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "UNBINNED"
    if not bins:
        return "UNBINNED"

    lo0, hi0 = bins[0]
    if value >= lo0 and value <= hi0:
        return f"({float(f'{lo0:.3f}')} - {float(f'{hi0:.3f}')}]"

    for lo, hi in bins[1:]:
        if value > lo and value <= hi:
            return f"({float(f'{lo:.3f}')} - {float(f'{hi:.3f}')}]"

    return "UNBINNED"


def coerce_esa(x) -> Optional[float]:
    """
    Generic float coercion (name kept for backward-compat with imports).
    Returns float(x) if finite, else None.
    """
    if x is None:
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _valid_floats(values: Iterable) -> List[float]:
    out: List[float] = []
    for s in values:
        v = coerce_esa(s)
        if v is not None:
            out.append(v)
    return out


def quantile_bin_tuples(scores: Iterable, q: int = 3) -> List[Tuple[float, float]]:
    """
    Return quantile cut bins as a list of (lo, hi) tuples over the data.

    - scores: iterable of numeric-like values (None/invalid are ignored)
    - q: desired number of quantiles (e.g., 3 for terciles)
    - Works for arbitrary ranges (including negatives).
    - Ensures strictly increasing edges; may yield fewer than q bins if edges collapse.
    """
    vals = _valid_floats(scores)
    if not vals:
        return []

    try:
        _, edges = pd.qcut(pd.Series(vals), q=q, retbins=True, duplicates="drop")
    except Exception:
        # Fallback to equal-width bins if qcut fails (e.g., too few unique values)
        return equal_width_bins(vals, k=q)

    uniq: List[float] = [float(edges[0])]
    for e in edges[1:]:
        e = float(e)
        if e > uniq[-1]:
            uniq.append(e)

    if len(uniq) < 2:
        return []

    return [(uniq[i], uniq[i + 1]) for i in range(len(uniq) - 1)]


def equal_width_bins(scores: Iterable, k: int = 3) -> List[Tuple[float, float]]:
    """
    Build k equal-width bins over the observed range of scores (supports negatives).
    Returns [] if the range is degenerate.

    Example: scores in [-1.5, 0.2, 3.7], k=3
      -> [(-1.5, 0.6333...), (0.6333..., 2.7666...), (2.7666..., 3.7)]
    """
    vals = _valid_floats(scores)
    if not vals:
        return []

    lo = min(vals)
    hi = max(vals)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [(lo, hi)]

    width = (hi - lo) / float(k)
    edges = [lo + i * width for i in range(k)] + [hi]

    uniq: List[float] = [edges[0]]
    for e in edges[1:]:
        e = float(e)
        if e > uniq[-1]:
            uniq.append(e)
    if len(uniq) < 2:
        return []

    return [(uniq[i], uniq[i + 1]) for i in range(len(uniq) - 1)]


def balance_bins(bin_tuples: Sequence[Tuple[float, float]], items: List[dict], seed: int = 42) -> None:
    """
    Downsample items per bin to the size of the smallest bin.

    Mutates `items` in place by setting item["difficulty"] = "UNBINNED" for
    overflow items that are randomly pruned from their bin. Items are not removed
    from the list; only their difficulty is marked.

    - bin_tuples: list of (lo, hi) numeric tuples
    - items: list of dicts; expects 'difficulty_score' either at top level
             or in item['meta']['difficulty_score']
    - seed: RNG seed for reproducible shuffling
    """
    if not bin_tuples:
        return

    buckets: List[List[dict]] = [[] for _ in range(len(bin_tuples))]

    # Assign items to buckets based on difficulty score
    for it in items:
        score = it.get("difficulty_score")
        if score is None:
            meta = it.get("meta", {}) if isinstance(it.get("meta", {}), dict) else {}
            score = meta.get("difficulty_score")
        score = coerce_esa(score)
        if score is None:
            continue

        for i, (lo, hi) in enumerate(bin_tuples):
            if lo <= score <= hi:
                buckets[i].append(it)
                break

    non_empty_sizes = [len(b) for b in buckets if len(b) > 0]
    if not non_empty_sizes:
        return

    target = min(non_empty_sizes)

    rng = random.Random(seed)
    for bucket in buckets:
        rng.shuffle(bucket)
        while len(bucket) > target:
            bucket[-1]["difficulty"] = "UNBINNED"
            bucket.pop()