#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:25:47 2020

@author: samal
"""

import numpy as np
import scipy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt

from MlsSim import MlsSim

def one(points):
    return np.ones(len(points), dtype='float64')

def x(points):
    return points[:,0]

def y(points):
    return points[:,1]

def xpy(points):
    return points[:,0] + points[:,1]

def x2(points):
    return points[:,0]**2

def y2(points):
    return points[:,1]**2

def xy(points):
    return points[:,0] * points[:,1]

def x2y2(points):
    return points[:,0]**2 * points[:,1]**2

func = x2y2

mls = MlsSim(2, support=-1, form='cubic')

N = 64
points = ( np.indices((N+1, N+1), dtype='float64').T.reshape(-1,2) ) / N
indices = np.arange(mls.nNodes, dtype = 'uint32')

phis = np.empty((len(points), mls.nNodes), dtype='float64')

for i, point in enumerate(points):
    phis[i] = mls.shapeFunctions0(point, indices)

A = np.empty((mls.nNodes, mls.nNodes), dtype='float64')
for i, node in enumerate(mls.nodes):
    A[i] = mls.shapeFunctions0(node, indices)
b = func(mls.nodes)
u = la.solve(A,b)

approximate_function = phis@u
exact_function = func(points)

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(15,15)
mpl.rc('axes', titlesize='xx-large', labelsize='x-large')
mpl.rc('xtick', labelsize='large')
mpl.rc('ytick', labelsize='large')
plt.subplots_adjust(hspace = 0.3, wspace = 0.25)

# plot the result
plt.subplot(221)
plt.tripcolor(points[:,0], points[:,1], approximate_function, shading='gouraud')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('MLS Approximation')

# plot the result
plt.subplot(222)
plt.tripcolor(points[:,0], points[:,1], exact_function, shading='gouraud')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Exact Function')

# plot the result
plt.subplot(223)
plt.tripcolor(points[:,0], points[:,1], approximate_function - exact_function, shading='gouraud')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Difference')

# subplots = [337, 338, 339, 334, 335, 336, 331, 332, 333]

# for i in range(9):
#     # plot the result
#     plt.subplot(subplots[i])
#     plt.tripcolor(points[:,0], points[:,1], phis[:,i], shading='gouraud')
#     plt.colorbar()
#     plt.xlabel(r'$x$')
#     plt.ylabel(r'$y$')
#     plt.title(f'$\Phi_{i}$')