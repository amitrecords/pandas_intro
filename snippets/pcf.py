# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import cumtrapz


def pcf(A, B, a, twobody, dr=0.01, start=0.5, end=7.0):
    '''
    Pair correlation function between two atom types.
    '''
    distances = twobody.loc[(twobody['symbols'] == A + B) |
                            (twobody['symbols'] == B + A), 'distance'].values
    bins = np.arange(start, end, dr)
    bins = np.append(bins, bins[-1] + dr)
    hist, bins = np.histogram(distances, bins)
    n = len(distances)
    rho = n / a**3
    r = (bins[1:] + bins[:-1]) / 2
    r3 = bins[1:]**3 - bins[:-1]**3
    normalization = a**3 / (n * 4 / 3 * np.pi * r3)
    g = normalization * hist
    i = cumtrapz(g, dx=dr)
    i = np.insert(i, 0, 0.0)
    return pd.DataFrame.from_dict({'r': r, 'g': g, 'i': i})

