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
