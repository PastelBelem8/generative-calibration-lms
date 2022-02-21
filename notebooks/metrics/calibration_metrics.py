from collections import defaultdict

import numpy as np


def sq_error(x, y):
    return (y - x) ** 2


def calibration_error(x, y):
    return y - x


def equal_width_ece_bins(x, y, bins: int = 20):
    n = len(x)

    bins = np.linspace(0, 1, bins + 1)
    nnzero_bins = np.digitize(x, bins=bins)

    bins_xs = defaultdict(list)
    bins_ys = defaultdict(list)
    bins_len = defaultdict(lambda: 0)

    for x_i, y_i, b_i in zip(x, y, nnzero_bins):
        bins_xs[b_i].append(x_i)
        bins_ys[b_i].append(y_i)
        bins_len[b_i] += 1

    calib_errors = np.zeros_like(bins)
    for i, b in enumerate(bins):
        bin_x = bins_xs.get(i, [0])
        bin_y = bins_ys.get(i, [0])
        bin_len = bins_len.get(i, 0)

        avg_x = np.mean(bin_x)
        avg_y = np.mean(bin_y)
        calib_errors[i] = bin_len * np.abs(avg_x - avg_y)

    calib_errors = calib_errors / n
    return calib_errors


def equal_freq_ece(x, y, frac: float = 0.10, n: int = None):
    if frac is not None:
        assert 0 < frac < 1, f"Invalid frac value: {frac}"
        n = int(round(len(x) * frac, 0))
    elif n is not None:
        assert n > 0, f"Invalid n value: {n}"

    ix = np.argsort(x)
    x_sorted = x[ix]
    y_sorted = y[ix]

    calib_errors = []
    for i_start in range(0, len(x), n):
        i_end = min(i_start + n, len(x))

        avg_x = np.mean(x_sorted[i_start:i_end])
        avg_y = np.mean(y_sorted[i_start:i_end])

        bin_len = i_end - i_start
        calib_errors.append(bin_len * np.abs(avg_x - avg_y) / len(x))

    return calib_errors
