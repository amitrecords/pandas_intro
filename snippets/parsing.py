# -*- coding: utf-8 -*-
'''
Parsing solutions
'''
import pandas as pd
import numpy as np


def naive_xyz_parser(path):
    '''
    Simple xyz parser

    Args:
        path (str): String xyz file path

    Returns:
        data (list): List of lists of xyz trajectory rows (excluding comments and atoms)
    '''
    data = []
    with open(path) as f:
        for line in f:
            line_split = line.split(' ')
            if len(line_split) == 4:
                try:
                    float(line_split[1])     # Check that this is not a comment line
                    data.append([line_split[0], float(line_split[1]), float(line_split[2]), float(line_split[3])])
                except:
                    pass                     # Ignore the line if it is a comment
    return data


def pandas_xyz_parser(path):
    '''
    Parse xyz files using pandas read_csv.

    Args:
        path (str): XYZ file path

    Returns:
        df (:class:`pandas.DataFrame`): Table of XYZ data
    '''
    df = pd.read_csv(path, delim_whitespace=True, names=['symbol', 'x', 'y', 'z'])    # Read data from disk
    df.dropna(inplace=True)                                                           # Drop nat line
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(np.float)                        # Convert types
    return df


def parse(path, nframe, nat):
    '''
    Complete parsing of xyz files.
    '''
    df = pandas_xyz_parser(path)
    df.index = pd.MultiIndex.from_product((range(nframe), range(nat)), names=['frame', 'atom'])
    return df
