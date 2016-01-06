# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import cumtrapz


def pcf(A, B, a, twobody, dr=0.05, start=0.5, end=7.5):
    '''
    Pair correlation function between two atom types.
    '''
    selection = twobody.loc[(twobody['symbols'] == A + B) |
                            (twobody['symbols'] == B + A)]
    distances = selection['distance'].values
    bins = np.arange(start, end, dr)
    bins = np.append(bins, bins[-1] + dr)
    hist, bins = np.histogram(distances, bins)
    n = hist.sum()
    rho = n / a**3
    r = (bins[1:] + bins[:-1]) / 2
    r3 = bins[1:]**3 - bins[:-1]**3
    normalization = a**3 / (n * 4 / 3 * np.pi * r3)
    g = normalization * hist
    n = cumtrapz(g, x=r, dx=dr)
    n = np.append(n, np.NaN)
    n *= 4 / 3 * np.pi * r3
    return pd.DataFrame.from_dict({'r': r, 'g': g, 'n': n})
