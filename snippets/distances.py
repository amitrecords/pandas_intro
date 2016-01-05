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


def cubic_periodic_distances(xyz, a, nat, k, min_distance=0.05):
    '''
    Computes atom to atom distances for a periodic cubic cell.

    Args:
        xyz: Properly indexed pandas DataFrame
        a: Cubic cell dimension

    Returns:
        twobody: DataFrame of distances
    '''
    # Since the unit cell size doesn't change between frames,
    # lets put all of the atoms (in every frame) back in the
    # unit cell at the same time.
    unit_xyz = create_unit(xyz, a)
    # Now we will define another function which will do the
    # steps we outlined above (see below) and apply this
    # function to every frame of the unit_xyz
    twobody = unit_xyz.groupby(level='frame').apply(_compute, k=k, min_distance=min_distance)
    # Pair the symbols
    #twobody.loc[:, 'symbols'] = [''.join(sorted(x)) for x in zip(*(twobody['atom1'].tolist(), twobody['atom2'].tolist()))]
    # Name the indexes
    twobody.index.names = ['frame', 'two']
    return twobody


# Leading underscore to emphasize that this function
# should not be called directly
def _compute(unit_frame, k, min_distance, max_distance=25.0):
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
    frame_twobody = frame_twobody.loc[frame_twobody['distance'] > min_distance]
    return frame_twobody
