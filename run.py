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


splitter = '>='
pkgs = ['numpy>=1.10.0',
        'scipy>=0.16',
        'pandas>=0.17',
        'tables>=3.2.2',
        'jupyter>=1.0.0',
        'notebook>=4.0.6',
        'ipython>=4.0.0',
        'matplotlib>=1.5.0',
        'seaborn>=0.6.0']


def install_pkgs(using='pip'):
    '''
    pip install pkgs
    '''
    install = 'install' if using == 'pip' else 'install -y'
    for pkgv in pkgs:
        pkg, version = pkgv.split(splitter)
        subprocess.run([using, install, pkg])


def start_notebook():
    '''
    Starts a jupyter notebook server in the current directory
    '''
    subprocess.run(['jupyter', 'notebook'])


if __name__ == '__main__':
    response = input('This script will install some dependencies. Continue [pip/conda] (default: pip): ')
    if response is None or response == 'pip':
        install_pkgs()
    elif response == 'conda':
        install_pkgs('conda')
    else:
        raise Exception('Unknown option {0}'.format(response))
    start_notebook()

