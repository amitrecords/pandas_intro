#! /usr/bin/env python
'''
Introduction to pandas
=======================
Quick script to install dependencies and start the notebook server
in the current directory.
'''
import subprocess
try:
    hasattr(raw_input, '__call__')
    input = raw_input
except:
    pass


pkgs = ['numpy', 'scipy', 'pandas', 'jupyter',
        'notebook', 'ipython', 'matplotlib', 'seaborn',
        'numba']


def install_pkgs(using='pip'):
    '''
    pip install pkgs
    '''
    cmd = [using, 'install']
    if using == 'pip':
        cmd += pkgs + ['tables']
    else:
        cmd += ['-y'] + pkgs + ['pytables']
    subprocess.run(cmd)


def start_notebook():
    '''
    Starts a jupyter notebook server in the current directory
    '''
    subprocess.run(['jupyter', 'notebook'])


if __name__ == '__main__':
    response = input('This script will install some dependencies. Continue [pip/conda/anaconda] (default: pip): ')
    if response is '' or response == 'pip':
        install_pkgs()
    elif response == 'conda':
        install_pkgs('conda')
    elif response == 'anaconda':
        install_pkgs('anaconda')
    else:
        raise Exception('Unknown option {0}'.format(response))
    start_notebook()

