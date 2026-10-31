"""Small statistics helpers."""

DEFAULT_PRECISION = 2


def mean(values):
    """Arithmetic mean of a non-empty sequence."""
    if len(values) == 0:
        raise ValueError("mean() of empty sequence")
    return sum(values) / len(values)


def _sorted_copy(values):
    return sorted(list(values))


def median(values):
    """Middle value; average of the two middle values for even-length input."""
    data = _sorted_copy(values)
    n = len(data)
    if n == 0:
        raise ValueError("median() of empty sequence")
    mid = n // 2
    if n % 2 == 1:
        return data[mid]
    return (data[mid - 1] + data[mid]) / 2


def variance(values):
    """Population variance."""
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)


def clamp(x, low, high):
    """Limit x to the closed interval [low, high]."""
    if low > high:
        raise ValueError("low must be <= high")
    if x < low:
        return low
    if x > high:
        return high
    return x


def rounded_mean(values, precision=DEFAULT_PRECISION):
    return round(mean(values), precision)
