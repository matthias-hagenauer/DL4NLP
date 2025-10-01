import math
import random
import pandas as pd

def parse_bins(spec):
    """
    Parse a string like "0-30,30-60,60-100" into a list of (lo, hi) float tuples.
    """
    bins = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            lo_s, hi_s = part.split("-")
            lo, hi = float(lo_s), float(hi_s)
        except Exception:
            raise ValueError("Invalid bin segment '%s' in spec '%s'" % (part, spec))
        if hi <= lo:
            raise ValueError("Invalid bin '%s' (requires hi > lo)" % part)
        bins.append((lo, hi))
    return bins


def assign_bin(value, bins):
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
        return f"({lo0}-{hi0}]"

    for lo, hi in bins[1:]:
        if value > lo and value <= hi:
            return f"({lo}-{hi}]"

    return "UNBINNED"


def coerce_esa(x):
    """
    Convert input to float if possible, otherwise return None.
    """
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def quantile_bin_tuples(scores, q=3):
    """
    Return quantile cut bins as a list of (lo, hi) tuples.

    - scores: iterable of numeric-like values (None/invalid are ignored)
    - q: number of quantiles desired (e.g., 3 for terciles)
    - Edges are clamped to [0, 100] and rounded to integers.
    - Ensures strictly increasing edges after rounding; may yield fewer than q bins.
    """
    vals = []
    for s in scores:
        try:
            if s is not None:
                vals.append(float(s))
        except Exception:
            pass
    if not vals:
        return []

    # qcut may drop duplicate edges when data has many ties
    _, edges = pd.qcut(pd.Series(vals), q=q, retbins=True, duplicates="drop")

    # Clamp outer range and round to integers
    edges[0] = 0
    edges[-1] = 100
    edges = [round(float(e)) for e in edges]

    # Ensure strictly increasing edges after rounding
    uniq = [edges[0]]
    for e in edges[1:]:
        if e > uniq[-1]:
            uniq.append(e)

    if len(uniq) < 2:
        return []

    return [(uniq[i], uniq[i + 1]) for i in range(len(uniq) - 1)]


def balance_bins(bin_tuples, items, seed=42):
    """
    Downsample items per bin to the size of the smallest bin.

    Mutates `items` in place by setting item["difficulty"] = "UNBINNED" for
    overflow items that are randomly pruned from their bin. Items are not removed
    from the list; only their difficulty is marked.

    - bin_tuples: list of (lo, hi) numeric tuples
    - items: list of dicts; expects 'esa_score' either at top level or in item['meta']['esa_score']
    - seed: RNG seed for reproducible shuffling
    """
    if not bin_tuples:
        return

    buckets = [[] for _ in range(len(bin_tuples))]

    # Assign items to buckets based on ESA score
    for it in items:
        score = it.get("esa_score")
        if score is None:
            meta = it.get("meta", {}) if isinstance(it.get("meta", {}), dict) else {}
            score = meta.get("esa_score")
        score = coerce_esa(score)
        if score is None:
            continue

        for i, (lo, hi) in enumerate(bin_tuples):
            if lo <= score <= hi:
                buckets[i].append(it)
                break

    # If all buckets are empty, nothing to balance
    non_empty_sizes = [len(b) for b in buckets if len(b) > 0]
    if not non_empty_sizes:
        return

    target = min(non_empty_sizes)

    rng = random.Random(seed)
    for bucket in buckets:
        rng.shuffle(bucket)
        # Mark overflow as UNBINNED and drop from the bucket view
        while len(bucket) > target:
            bucket[-1]["difficulty"] = "UNBINNED"
            bucket.pop()
