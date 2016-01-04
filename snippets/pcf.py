import pandas as pd


def pcf(A, B, a, twobody, dr=0.05, start=0.5, end=12.5):
    '''
    Pair correlation function between two atom types.
    '''
    symbols = ''.join(sorted((A, B)))
    distances = twobody.loc[twobody['symbols'] == symbols, 'distance']
    bins = np.arange(start, end, dr)
    bins = np.append(bins, bins[-1] + dr)
    hist, bins = np.histogram(distances, bins)
    n = len(distances)
    m = len(twobody.index.get_level_values('frame').unique())
    vol = a**3
    rho = n / vol
    r = (bins[1:] + bins[:-1]) / 2
    r3 = bins[1:]**3 - bins[:-1]**3
    denom = rho * 4 / 3 * np.pi * r3
    g = hist / denom
    i = np.cumsum(hist) / m
    return pd.DataFrame.from_dict({'r': r, 'g': g, 'i': i})
