from random import seed
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
    Assign a numeric value to a bin. Returns a string label or "UNBINNED".
    """
    if value is None:
        return "UNBINNED"
    for lo, hi in bins:
        if value >= lo and value <= hi:
            return "(%s-%s]" % (lo, hi)
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
    scores: iterable of numeric (or string) values.
    q: number of quantiles desired.
    round_digits: decimals to round edges (None to skip rounding).
    force_range: (lo, hi) to override first and last edge (e.g. (0,100)).
    """
    scores = [int(s) for s in scores]
    _, edges = pd.qcut(pd.Series(scores), q=q, retbins=True, duplicates='drop')
    edges[0] = 0
    edges[-1] = 100
    edges = [round(e) for e in edges]
    return [(edges[i], edges[i+1]) for i in range(len(edges)-1)]


def balance_bins(bin_tuples, items, seed=42):
    mega_bin = [[] for _ in range(len(bin_tuples))]
    for it in items:
        score = it.get("esa_score")
        for i, (lo, hi) in enumerate(bin_tuples):
            if lo <= score <= hi:
                mega_bin[i].append(it)
                break

        min_size = min([len(bin) for bin in mega_bin])
        import random
        random.seed(seed)
        for bin in mega_bin:
            random.shuffle(bin)
            ### Do random sampling and those that don't get sampled should be "UNBINNED"
            while len(bin) < min_size:
                bin[-1]["difficulty"] = "UNBINNED"
                bin.pop()
