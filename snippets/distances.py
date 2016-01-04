# -*- coding: utf-8 -*-
'''
Parsing solutions
'''
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from itertools import combinations
from numba import jit, float64, int64

def free_boundary_distances(frame):
    '''
    Compute all of the atom to atom distances with free boundary conditions
    '''
    xyz = frame.loc[:, ['x', 'y', 'z']]
    distances = pdist(xyz)                                          # Compute distances
    symbol = frame.loc[:, 'symbol']
    symbols = [''.join(syms) for syms in combinations(symbol, 2)]   # Compute symbols
    return pd.DataFrame.from_dict({'distances': distances, 'symbols': symbols})


def unitify(df, a):
    '''
    Put all atoms back into the cubic unit cell.
    '''
    unitdf = df.copy()              # Don't alter the original
    cell_dim = np.array([a, a, a])
    unitdf.loc[:, ['x', 'y', 'z']] = np.mod(unitdf.loc[:, ['x', 'y', 'z']], cell_dim)
    return unitdf


def superframe(frame, a):
    '''
    Generate a 3x3x3 supercell from a frame.
    '''
    v = [-1, 0, 1]
    n = len(frame)
    unit = frame.loc[:, ['x', 'y', 'z']].values
    coords = np.empty((n * 27, 3))
    h = 0
    for i in v:
        for j in v:
            for k in v:
                for l in range(n):
                    coords[h, 0] = unit[l, 0] + i * a
                    coords[h, 1] = unit[l, 1] + j * a
                    coords[h, 2] = unit[l, 2] + k * a
                    h += 1
    return coords


@jit(nopython=True)
def superframe_numba(unit, a):
    v = [-1, 0, 1]
    n = len(unit)
    coords = np.empty((n * 27, 3), dtype=float64)
    h = 0
    for i in v:
        for j in v:
            for k in v:
                for l in range(n):
                    coords[h, 0] = unit[l, 0] + i * a
                    coords[h, 1] = unit[l, 1] + j * a
                    coords[h, 2] = unit[l, 2] + k * a
                    h += 1
    return coords


def map_x_to_y(x, y):
    '''
    Using the indexes in x, generate an array of the same
    length populated by values from y.
    '''
    mapped = np.empty((len(x), ), dtype=np.int)
    for i, index in enumerate(x):
        mapped[i] = y[index]
    return mapped


@jit(nopython=True)
def map_x_to_y_numba(x, y):
    '''
    Using the indexes in x, generate an array of the same
    length populated by values from y.
    '''
    mapped = np.empty((len(x), ), dtype=int64)
    for i, index in enumerate(x):
        mapped[i] = y[index]
    return mapped


