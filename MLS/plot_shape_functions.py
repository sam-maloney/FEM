#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:25:47 2020

@author: samal
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from MlsSim import MlsSim

mls = MlsSim(2, support=-1, form='cubic')

N = 64
points = ( np.indices((N+1, N+1), dtype='float64').T.reshape(-1,2) ) / N
indices = np.arange(mls.nNodes, dtype = 'uint32')

phis = np.empty((len(points), mls.nNodes), dtype='float64')

for i, point in enumerate(points):
    phis[i] = mls.shapeFunctions0(point, indices)

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(15,15)
mpl.rc('axes', titlesize='xx-large', labelsize='x-large')
mpl.rc('xtick', labelsize='large')
mpl.rc('ytick', labelsize='large')
plt.subplots_adjust(hspace = 0.3, wspace = 0.25)

subplots = [337, 338, 339, 334, 335, 336, 331, 332, 333]

for i in range(9):
    # plot the result
    plt.subplot(subplots[i])
    plt.tripcolor(points[:,0], points[:,1], phis[:,i], shading='gouraud')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(f'$\Phi_{i}$')