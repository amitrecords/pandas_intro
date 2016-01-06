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


def create_unit(df, a):
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


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
def map_x_to_y_numba(x, y):
    '''
    Using the indexes in x, generate an array of the same
    length populated by values from y.
    '''
    mapped = np.empty((len(x), ), dtype=int64)
    for i, index in enumerate(x):
        mapped[i] = y[index]
    return mapped


def custom_pdist(origxyz, superxyz):
    n1 = len(origxyz)
    n2 = len(superxyz)
    distances = np.empty((n1 * n2, ))
    index1 = np.empty((n1 * n2), dtype='i8')
    index2 = np.copy(index1)
    h = 0
    for i, cen in enumerate(central):
        for j, b in enumerate(big):
            distances[h] = np.sum((cen - b)**2)**0.5    # Numpy broadcasting
            index1[h] = i
            index2[h] = j
            h += 1
    return distances, index1, index2


@jit(nopython=True, cache=True)
def custom_pdist_numba(origxyz, superxyz):
    n = len(origxyz)
    nn = len(superxyz)
    distances = np.empty((n * nn, ), dtype=float64)
    index1 = np.empty((n * nn, ), dtype=int64)
    index2 = np.empty((n * nn, ), dtype=int64)
    h = 0
    for i in range(n):
        for j in range(nn):
            csum = 0.0
            for k, c in enumerate(origxyz[i]):
                csum += (c - superxyz[j][k])**2
            distances[h] = csum**0.5
            index1[h] = i
            index2[h] = j
            h += 1
    return distances, index1, index2


def cubic_periodic_distances(xyz, a, nat, k=None):
    '''
    Computes atom to atom distances for a periodic cubic cell.

    Args:
        xyz: Properly indexed pandas DataFrame
        a: Cubic cell dimension

    Returns:
        twobody: DataFrame of distances
    '''
    k = nat - 1 if k is None else k
    # Since the unit cell size doesn't change between frames,
    # let's put all of the atoms (in every frame) back in the
    # unit cell at the same time.
    unit_xyz = create_unit(xyz, a)
    # Now we will define another function which will do the
    # steps we outlined above (see below) and apply this
    # function to every frame of the unit_xyz
    twobody = unit_xyz.groupby(level='frame').apply(_compute, k=k)    # <== This is the meat and potatoes
    # Filter the meaningful distances
    twobody = twobody.loc[(twobody.distance > 0.3) & (twobody.distance < 8.3)]
    # Pair the symbols
    twobody.loc[:, 'symbols'] = twobody['atom1'] + twobody['atom2']
    # Name the indices
    twobody.index.names = ['frame', 'two']
    return twobody


def _compute(unit_frame, k):
    '''
    Compute periodic atom to atom distances
    '''
    # Generate superframe
    values = unit_frame.loc[:, ['x', 'y', 'z']].values
    big_frame = superframe_numba(values, a)

    # Create the K-D tree
    kd = cKDTree(big_frame)
    distances, indexes = kd.query(values, k=k)

    # Metadata
    unit_frame_indexes = np.tile(unit_frame.index.get_level_values('atom').values, (len(values), 27)).flatten()
    symbol_dict = unit_frame.reset_index('frame', drop=True).loc[:, 'symbol'].to_dict()
    repeated_source = np.repeat(indexes[:, 0], k)
    def symbol_caller(symbol):
        return symbol_dict[symbol]

    # Mapping of atom indexes to symbols
    atom1_indexes = map_x_to_y_numba(repeated_source, unit_frame_indexes)
    atom2_indexes = map_x_to_y_numba(indexes.flatten(), unit_frame_indexes)
    atom1_symbols = list(map(symbol_caller, atom1_indexes))
    atom2_symbols = list(map(symbol_caller, atom2_indexes))

    # Generation of the DataFrame
    frame_twobody = pd.DataFrame.from_dict({'distance': distances.flatten(), 'atom1': atom1_symbols, 'atom2': atom2_symbols})
    return frame_twobody
