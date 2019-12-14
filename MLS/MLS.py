#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Meshfree method simulation using moving less squares (MLS)

@author: Sam Maloney
"""

import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt

import scipy.linalg as la

class MlsSim(object):
    """Class for meshless moving least squares (MLS) method."""
    
    def __init__(self, N, Nquad=1, support=-1, form='cubic'):
        self.N = N
        self.Nquad = Nquad
        if support > 0:
            self.support = support
        else: # if support is negative, set to default 1.5X grid spacing
            self.support = 1.5/(N)
        self.selectSpline(form)
        self.nodes = np.indices((N+1, N+1)).T.reshape(-1,2) / N
        self.generateQuadraturePoints(N, Nquad)
        # self.quads = np.empty((N*N*Nquad*Nquad, 2), dtype='float64')
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.N},{self.Nquad}," \
               f"{self.support},'{self.form}')"
    
    def generateQuadraturePoints(self, N, Nquad):
        self.quads = (np.indices((N, N)).T.reshape(-1,2) + 0.5) / N
    
    def cubicSpline(self, r):
        """Compute cubic spline function and its radial derivative."""
        if r < 0.5:
            w = 2.0/3.0 - 4.0*r*r + 4.0*r*r*r
            dwdr = -8.0*r + 12.0*r*r
        elif r < 1.0:
            w = 4.0/3.0 - 4.0*r + 4.0*r*r - 4.0/3.0*r*r*r
            dwdr = -4.0 + 8.0*r - 4.0*r*r
        else:
            w = 0.0
            dwdr = 0.0
        return w, dwdr
    
    def quarticSpline(self, r):
        """Compute quartic spline function and its radial derivative."""
        if r < 1.0:
            w = 1.0 - 6.0*r*r + 8.0*r*r*r - 3.0*r*r*r*r
            dwdr = -12.0*r + 24.0*r*r - 12.0*r*r*r
        else:
            w = 0.0
            dwdr = 0.0
        return w, dwdr
    
    def selectSpline(self, form):
        """Register the spline method to the correct order computation."""
        if form == 'cubic':
            self.spline = self.cubicSpline
            self.form = form
        elif form == 'quartic':
            self.spline = self.quarticSpline
            self.form = form
        else:
            print(f"Error: unkown spline form '{form}'. "
                  f"Must be one of 'cubic' or 'quartic'.")
    
    def weightFunction(self, quadPoint, nodePoint):
        """Compute weight function and gradient using cubic or quartic spline.
        
        Arguments:
        quadPoint : array([x,y]) coordinates of point at which w is evaluated;
        nodePoint : array([x,y]) coordinates of node I.
        
        Returns:
        w     : value of the weight function w at the quadrature point;
        gradw : array([dx,dy]) gradient of w at the quadrature point.
        """
        displacement = (quadPoint - nodePoint)/self.support
        distance = la.norm(displacement)
        w, dwdr = self.spline(distance)
        gradr = displacement/(distance*self.support)
        gradw = dwdr * gradr
        return w, gradw

    def shapeFunctions(self, quadPoint, indices):
        """Compute shape function at quad point for all nodes in its support.
        Basis used is linear basis pT = [1 x y].
        
        Arguments:
        quadPoint : array([x,y]) coordinate of quadrature point;
        indices   : (1D array) indices of nodes within support of pt.
        
        Returns:
        phi     : (1D array) values of phi for all nodes in indices;
        gradphi : (nx2 array) gradients of phi for all n nodes in indices.
        """
        # --------------------------------------
        #     compute the moment matrix A(x)
        # --------------------------------------
        A = np.zeros((3,3), dtype='float64')
        dAdx = np.zeros((3,3), dtype='float64')
        dAdy = np.zeros((3,3), dtype='float64')
        w = np.empty((len(indices)), dtype='float64')
        gradw = np.empty((len(indices), 2), dtype='float64')
        for i, index in enumerate(indices):
            wi, gradwi = self.weightFunction(quadPoint, self.nodes[index,:].flatten())
            pTp = np.outer( [1.0, self.nodes[index,0], self.nodes[index,1]],
                            [1.0, self.nodes[index,0], self.nodes[index,1]] )
            A = A + wi*pTp
            dAdx = dAdx + gradwi[0]*pTp
            dAdy = dAdy + gradwi[1]*pTp
            w[i] = wi
            gradw[i,:] = gradw
        # --------------------------------------
        #         compute  matrix c(x)
        # --------------------------------------
        # A(x)c(x)   = p(x)
        # A(x)c_k(x) = b_k(x)
        # Backward substitutions, once for c(x), twice for c_k(x) k=1,2
        # Using LU factorization for A
        p = np.concatenate(([1.0], quadPoint))
        lu, piv = la.lu_factor(A, check_finite=True)
        c = np.empty((3,3), dtype='float64')
        c[:,0] = la.lu_solve((lu, piv), piv.T @ p,
                             check_finite=True)
        c[:,1] = la.lu_solve((lu, piv), piv.T @ ([[0],[1],[0]] - dAdx@c[:,0:1]),
                             check_finite=True)
        c[:,2] = la.lu_solve((lu, piv), piv.T @ ([[0],[0],[1]] - dAdy@c[:,0:1]),
                             check_finite=True)
        # --------------------------------------
        #       compute phi and gradphi
        # --------------------------------------
        phi = np.empty((len(indices)), dtype='float64')
        gradphi = np.empty((len(indices), 2), dtype='float64')
        for i, index in enumerate(indices):
            pi = np.concatenate(([1.0], self.nodes[index,:].flatten()))
            cpi = c[:,0] @ pi
            phi[i] = cpi * w[i]
            gradphi[i,:] = (c[:,1:3].T @ pi * w[i]).T + cpi*gradw[i]
        return phi, gradphi
        
    def defineSupport(self, quadPoint):
        """Find nodes within support of a given quadrature point.
        
        Argument:
        quadPoint : array([x,y]) coordinates of quadrature point.
        
        Return:
        indices : (1D array) of indices of nodes within support of quad point.
        """
        distances = la.norm(quadPoint - self.nodes, axis=1)
        indices = np.argwhere(distances < self.support).flatten()
        return indices