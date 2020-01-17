#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:20:15 2020

@author: samal
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la


class MlsSim(object):
    """Class for meshless moving least squares (MLS) method."""
    
    def __init__(self, N, k=1, Nquad=1, support=-1, form='cubic',
                 method='galerkin'):
        self.N = N
        self.k = k
        self.nCells = N*N
        self.nNodes = (N+1)*(N+1)
        self.Nquad = Nquad
        if support > 0:
            self.support = support
        else: # if support is negative, set to default 1.4X grid spacing
            self.support = 1.4/(N)
        self.nodes = ( np.indices((N+1, N+1), dtype='float64')
                       .T.reshape(-1,2) ) / N
        self.isBoundaryNode = np.any(np.mod(self.nodes, 1) == 0, axis=1)
        self.nBoundaryNodes = np.count_nonzero(self.isBoundaryNode)
        # self.g = ( np.sin (k*np.pi*self.nodes[self.isBoundaryNode,0])
        #          * np.sinh(k*np.pi*self.nodes[self.isBoundaryNode,1]) ).round(
        #            decimals=14)
        self.g = self.nodes[self.isBoundaryNode,0] * self.nodes[self.isBoundaryNode,1]
        self.selectSpline(form)
        self.selectMethod(method)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.N},{self.k}," \
               f"{self.Nquad},{self.support},'{self.form}','{self.method}')"
    
    def selectSpline(self, form):
        """Register the 'self.spline' method to the correct order computation.
        
        Parameters
        ----------
        form : string
            Must be either 'cubic' or 'quartic'.

        Returns
        -------
        None.

        """
        if form == 'cubic':
            self.spline = self.cubicSpline
            self.form = form
        elif form == 'quartic':
            self.spline = self.quarticSpline
            self.form = form
        else:
            print(f"Error: unkown spline form '{form}'. "
                  f"Must be one of 'cubic' or 'quartic'.")
    
    def selectMethod(self, method):
        """Register the 'self.assembleStiffnesMatrix' method.
        
        Parameters
        ----------
        method : string
            Must be either 'galerkin' or 'collocation'.
        Nquad : integer in [1,2]
            Number of quadrature points in each grid cell along one dimension.

        Returns
        -------
        None.

        """
        if method == 'galerkin':
            self.assembleStiffnessMatrix = self.assembleGalerkinStiffnessMatrix
            self.method = method
            self.generateQuadraturePoints(self.N, self.Nquad)
            self.b = np.concatenate((np.zeros(self.nNodes,dtype='float64'),
                                     self.g))
        elif method == 'collocation':
            self.assembleStiffnessMatrix = self.assembleCollocationStiffnessMatrix
            self.method = method
            self.b = np.zeros(self.nNodes,dtype='float64')
            self.b[self.isBoundaryNode] = self.g
        else:
            print(f"Error: unkown assembly method '{method}'. "
                  f"Must be one of 'galerkin' or 'collocation'.")
    
    def generateQuadraturePoints(self, N, Nquad):
        """Compute list of (x,y) quadrature points for Galerkin integration.

        Parameters
        ----------
        N : integer
            Number of grid cells along one dimension.
        Nquad : integer in [1,2]
            Number of quadrature points in each grid cell along one dimension.

        Returns
        -------
        None.

        """
        if Nquad == 1:
            self.quads = ( np.indices((N, N), dtype='float64')
                           .T.reshape(-1,2) + 0.5 ) / N
            self.quadWeight = 1.0/(N*N)
            self.nQuads = len(self.quads)
        elif Nquad == 2:
            tmp = ( np.indices((N, N), dtype='float64')
                           .T.reshape(-1,2) + 0.5 ) / N
            # offset = 0.288675134594812882254574390251/N # 0.5/sqrt(3)/N
            offset = 0.25/N
            self.quads = np.concatenate( (
                tmp - offset,
                tmp + offset,
                np.hstack((tmp[:,0:1] + offset, tmp[:,1:2] - offset)),
                np.hstack((tmp[:,0:1] - offset, tmp[:,1:2] + offset))))
            self.quadWeight = 0.25/(N*N)
            self.nQuads = len(self.quads)
        else:
            print(f"Error: bad Nquad value of '{Nquad}'. "
                  f"Must be either 1 or 2.")
    
    def cubicSpline(self, r):
        """Compute cubic spline function and its radial derivative.

        Parameters
        ----------
        r : float
            Distance from evaluation point to node point.

        Returns
        -------
        w : float
            Value of the cubic spline function at the given distance.
        dwdr : float
            Value of the radial derivative at the given distance.
        d2wdr2 : float
            Value of the 2nd order radial derivative at the given distance.

        """
        r2 = r*r
        r3 = r2*r
        if r < 0.5:
            w = 2.0/3.0 - 4.0*r2 + 4.0*r3
            dwdr = -8.0*r + 12.0*r2
            d2wdr2 = -8.0 + 24.0*r
        elif r < 1.0:
            w = 4.0/3.0 - 4.0*r + 4.0*r2 - 4.0/3.0*r3
            dwdr = -4.0 + 8.0*r - 4.0*r2
            d2wdr2 = 8.0 - 8.0*r
        else:
            w = 0.0
            dwdr = 0.0
            d2wdr2 = 0.0
        return w, dwdr, d2wdr2
    
    def quarticSpline(self, r):
        """Compute quartic spline function and its radial derivative.

        Parameters
        ----------
        r : float
            Distance from evaluation point to node point.

        Returns
        -------
        w : float
            Value of the quartic spline function at the given distance.
        dwdr : float
            Value of the radial derivative at the given distance.
        d2wdr2 : float
            Value of the 2nd order radial derivative at the given distance.

        """
        if r < 1.0:
            r2 = r*r
            r3 = r2*r
            r4 = r2*r2
            w = 1.0 - 6.0*r2 + 8.0*r3 - 3.0*r4
            dwdr = -12.0*r + 24.0*r2 - 12.0*r3
            d2wdr2 = -12.0 + 48.0*r - 36.0*r2
        else:
            w = 0.0
            dwdr = 0.0
            d2wdr2 = 0.0
        return w, dwdr, d2wdr2
    
    def weightFunction(self, point, nodePoint):
        """Compute weight function and gradient using cubic or quartic spline.        

        Parameters
        ----------
        point : numpy.array([x,y], dtype='float64')
            Coordinates of evaluation point at which w is evaluated.
        nodePoint : numpy.array([x,y], dtype='float64')
            Coordinates of node I.

        Returns
        -------
        w : float
            Value of the weight function at the evaluation point.
        gradw : numpy.array([dx,dy], dtype='float64')
            Gradient of weight function at the evaluation point.
        grad2w : numpy.array([dxx,dyy], dtype='float64')
            2nd order derivatives of weight function at the evaluation point.

        """
        displacement = (point - nodePoint)/self.support
        distance = la.norm(displacement)
        w, dwdr, d2wdr2 = self.spline(distance)
        if distance > 1e-14:
            gradr = displacement/(distance*self.support)
            gradw = dwdr * gradr
            grad2w = d2wdr2 * gradr*gradr
        else:
            gradw = np.zeros(2,dtype='float64')
            gradr = np.repeat(1.0/(np.sqrt(2)*self.support), 2)
            grad2w = d2wdr2 * gradr*gradr
        return w, gradw, grad2w
    
    def shapeFunctions0(self, point, indices, check = False):
        """Compute shape function at quad point for all nodes in its support.
        Basis used is linear basis pT = [1 x y].
        Computes the shape function value and its gradient.

        Parameters
        ----------
        point : numpy.array([x,y], dtype='float64')
            Coordinate of evaluation point.
        indices : numpy.array([...], dtype='uint32')
            Indices of nodes within support of given evaluation point.
        check : boolean, optional
            Passed as 'check_finite' to lu_factor and lu_solve of scipy.linalg.
            The default is False.

        Returns
        -------
        phi : numpy.array([...], dtype='float64')
            Values of phi for all nodes in indices.

        """
        # --------------------------------------
        #     compute the moment matrix A(x)
        # --------------------------------------
        A = np.zeros((3,3), dtype='float64')
        w = np.empty((len(indices)), dtype='float64')
        p = np.empty((len(indices), 3), dtype='float64')
        for i, node in enumerate(self.nodes[indices]):
            wi = self.weightFunction(point, node)[0]
            pi = np.concatenate(([1.0], node))
            pTp = np.outer(pi, pi)
            A += wi*pTp
            w[i] = wi
            p[i] = pi
        # --------------------------------------
        #      compute  matrix c(x) and phi
        # --------------------------------------
        # A(x)c(x)   = p(x)
        # Backward substitutions, once for c(x) using LU factorization for A
        p_x = np.concatenate(([1.0], point))
        lu, piv = la.lu_factor(A, overwrite_a=True, check_finite=check)
        c = la.lu_solve((lu, piv), p_x, overwrite_b=True, check_finite=check)
        phi = c @ p.T * w
        return phi

    def shapeFunctions1(self, point, indices, check = False):
        """Compute shape function at quad point for all nodes in its support.
        Basis used is linear basis pT = [1 x y].
        Computes the shape function value and its gradient.

        Parameters
        ----------
        point : numpy.array([x,y], dtype='float64')
            Coordinate of evaluation point.
        indices : numpy.array([...], dtype='uint32')
            Indices of nodes within support of given evaluation point.
        check : boolean, optional
            Passed as 'check_finite' to lu_factor and lu_solve of scipy.linalg.
            The default is False.

        Returns
        -------
        phi : numpy.array([...], dtype='float64')
            Values of phi for all nodes in indices.
        gradphi : nx2 ndarray, dtype='float64'
            Gradients of phi for all nodes in indices. [[dx1,dy1],[dx2,dy2]...]

        """
        # --------------------------------------
        #     compute the moment matrix A(x)
        # --------------------------------------
        A = np.zeros((3,3), dtype='float64')
        dAdx = np.zeros((3,3), dtype='float64')
        dAdy = np.zeros((3,3), dtype='float64')
        w = np.empty((len(indices)), dtype='float64')
        p = np.empty((len(indices), 3), dtype='float64')
        gradw = np.empty((len(indices), 2), dtype='float64')
        for i, node in enumerate(self.nodes[indices]):
            wi, gradwi = self.weightFunction(point, node)[0:2]
            pi = np.concatenate(([1.0], node))
            pTp = np.outer(pi, pi)
            A += wi*pTp
            dAdx += gradwi[0]*pTp
            dAdy += gradwi[1]*pTp
            w[i] = wi
            p[i] = pi
            gradw[i] = gradwi
        # --------------------------------------
        #         compute  matrix c(x)
        # --------------------------------------
        # A(x)c(x)   = p(x)
        # A(x)c_k(x) = b_k(x)
        # Backward substitutions, once for c(x), twice for c_k(x) k=1,2
        # Using LU factorization for A
        p_x = np.concatenate(([1.0], point))
        lu, piv = la.lu_factor(A, check_finite=check)
        c = np.empty((3,3), dtype='float64')
        c[0] = la.lu_solve((lu, piv), p_x, check_finite=check)
        c[1] = la.lu_solve((lu, piv),([0,1,0] - dAdx@c[0]), check_finite=check)
        c[2] = la.lu_solve((lu, piv),([0,0,1] - dAdy@c[0]), check_finite=check)
        # --------------------------------------
        #       compute phi and gradphi
        # --------------------------------------
        cpi = c[0] @ p.T
        phi = cpi * w
        gradphi = ( c[1:3]@p.T*w + cpi*gradw.T).T
        return phi, gradphi
    
    def shapeFunctions2(self, point, indices, check = False):
        """Compute shape function at quad point for all nodes in its support.
        Basis used is linear basis pT = [1 x y].
        Computes the shape function value and its 2nd derivatives.

        Parameters
        ----------
        point : numpy.array([x,y], dtype='float64')
            Coordinate of evaluation point.
        indices : numpy.array([...], dtype='uint32')
            Indices of nodes within support of given evaluation point.
        check : boolean, optional
            Passed as 'check_finite' to lu_factor and lu_solve of scipy.linalg.
            The default is False.

        Returns
        -------
        phi : numpy.array([...], dtype='float64')
            Values of phi for all nodes in indices.
        grad2phi : nx2 ndarray, dtype='float64'
            2nd derivatives of phi for all nodes in indices.
            [[dxx1,dyy1],[dxx2,dyy2]...]

        """
        # --------------------------------------
        #     compute the moment matrix A(x)
        # --------------------------------------
        A = np.zeros((3,3), dtype='float64')
        dAdx = np.zeros((3,3), dtype='float64')
        dAdy = np.zeros((3,3), dtype='float64')
        d2Adx2 = np.zeros((3,3), dtype='float64')
        d2Ady2 = np.zeros((3,3), dtype='float64')
        w = np.empty((len(indices)), dtype='float64')
        gradw = np.empty((len(indices), 2), dtype='float64')
        grad2w = np.empty((len(indices), 2), dtype='float64')
        for i, node in enumerate(self.nodes[indices]):
            wi, gradwi, grad2wi = self.weightFunction(point, node)
            pTp = np.outer([1.0, node[0], node[1]], [1.0, node[0], node[1]])
            A += wi*pTp
            dAdx += gradwi[0]*pTp
            dAdy += gradwi[1]*pTp
            d2Adx2 += grad2wi[0]*pTp
            d2Ady2 += grad2wi[1]*pTp
            w[i] = wi
            gradw[i,:] = gradwi
            grad2w[i,:] = grad2wi
        # --------------------------------------
        #         compute  matrix c(x)
        # --------------------------------------
        # A(x)c(x)   = p(x)
        # A(x)c_k(x) = b_k(x)
        # Backward substitutions, once for c(x), twice for c_k(x) k=1,2
        # and twice for c_kk(x) k=1,2, using LU factorization for A
        p = np.concatenate(([1.0], point))
        lu, piv = la.lu_factor(A, check_finite=check)
        c = np.empty((5,3), dtype='float64')
        c[0] = la.lu_solve((lu, piv), p, check_finite=check)
        c[1] = la.lu_solve((lu, piv),([0,1,0] - dAdx@c[0]), check_finite=check)
        c[2] = la.lu_solve((lu, piv),([0,0,1] - dAdy@c[0]), check_finite=check)
        c[3] = la.lu_solve((lu, piv),(-2.0*dAdy@c[1] - d2Adx2@c[0]), check_finite=check)
        c[4] = la.lu_solve((lu, piv),(-2.0*dAdy@c[2] - d2Ady2@c[0]), check_finite=check)
        # --------------------------------------
        #       compute phi and gradphi
        # --------------------------------------
        phi = np.empty((len(indices)), dtype='float64')
        grad2phi = np.empty((len(indices), 2), dtype='float64')
        for i, node in enumerate(self.nodes[indices]):
            pi = np.concatenate(([1.0], node))
            cpi = c[0] @ pi
            phi[i] = cpi * w[i]
            grad2phi[i,:] = ( (c[3:5,:] @ pi * w[i]).T +
                             2.0*(c[1:3,:] @ pi * gradw[i]).T + cpi*grad2w[i] )
        return phi, grad2phi
        
    def defineSupport(self, point):
        """Find nodes within support of a given evaluation point.

        Parameters
        ----------
        point : numpy.array([x,y], dtype='float64')
            Coordinates of given evaluation point.

        Returns
        -------
        indices : numpy.array([...], dtype='uint32')
            Indices of nodes within support of given evaluation point.
            
        """
        distances = la.norm(point - self.nodes, axis=1)
        indices = np.flatnonzero(distances < self.support).astype('uint32')
        return indices
    
    def createKMatrix(self):
        """Initializes the sparse structure of the stiffness matrix K.

        Returns
        -------
        None.

        """
        saveSupport = self.support
        self.support *= 2.0
        # pre-allocate array for indices
        # this is the maximum possibly required size; not all will be used
        nMaxEntriesPerNode = int(self.nNodes*4*(self.support+0.25/self.N)**2)
        indices = np.empty(self.nNodes * nMaxEntriesPerNode, dtype='uint32')
        indptr = np.empty(self.nNodes+1, dtype='uint32')
        index = 0
        for iN, node in enumerate(self.nodes):
            inds = self.defineSupport(node)
            nEntries = len(inds)
            indptr[iN] = index
            indices[index:index+nEntries] = inds
            index += nEntries
        indptr[-1] = index
        self.K = sp.csr_matrix( (np.zeros(index), indices[0:index], indptr),
                                shape=(self.nNodes, self.nNodes) )
        self.support = saveSupport
    
    def assembleGalerkinStiffnessMatrix(self):
        """Assemble the Galerkin system stiffness matrix K in CSR format.

        Returns
        -------
        None.

        """
        # self.createKMatrix()
        # for iQ, quad in enumerate(self.quads):
        #     indices = self.defineSupport(quad)
        #     phi, gradphi = self.shapeFunctions1(quad, indices)
        #     self.K[np.repeat(indices, len(indices)), \
        #             np.tile(indices, len(indices))] += \
        #         np.ravel(gradphi@gradphi.T)
        
        # pre-allocate arrays for stiffness matrix triplets
        # these are the maximum possibly required sizes; not all will be used
        nMaxEntriesPerQuad = int((self.nNodes*4*(self.support+0.25/self.N)**2)**2)
        data = np.zeros(self.nQuads * nMaxEntriesPerQuad, dtype='float64')
        row_ind = np.zeros(self.nQuads * nMaxEntriesPerQuad, dtype='uint32')
        col_ind = np.zeros(self.nQuads * nMaxEntriesPerQuad, dtype='uint32')
        # build matrix for interior nodes
        index = 0
        for iQ, quad in enumerate(self.quads):
            indices = self.defineSupport(quad)
            nEntries = len(indices)**2
            phi, gradphi = self.shapeFunctions1(quad, indices)
            data[index:index+nEntries] = np.ravel(gradphi@gradphi.T)
            row_ind[index:index+nEntries] = np.repeat(indices, len(indices))
            col_ind[index:index+nEntries] = np.tile(indices, len(indices))
            index += nEntries
        inds = np.flatnonzero(data.round(decimals=14,out=data))
        # assemble the triplets into the sparse stiffness matrix
        self.K = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
                                shape=(self.nNodes, self.nNodes) )
        self.K *= self.quadWeight
        # apply Dirichlet boundary conditions using Lagrange multiplier method
        data.fill(0.0)
        row_ind.fill(0)
        col_ind.fill(0)
        index = 0
        for iN, node in enumerate(self.nodes[self.isBoundaryNode]):
            indices = self.defineSupport(node)
            nEntries = len(indices)
            phi = self.shapeFunctions0(node, indices)
            data[index:index+nEntries] = -1.0*phi
            row_ind[index:index+nEntries] = indices
            col_ind[index:index+nEntries] = np.repeat(iN, nEntries)
            index += nEntries
        inds = np.flatnonzero(data.round(decimals=14,out=data))
        G = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
                           shape=(self.nNodes, self.nBoundaryNodes) )
        G *= -1.0
        self.K = sp.bmat([[self.K, G], [G.T, None]], format='csr')
    
    def assembleCollocationStiffnessMatrix(self):
        """Assemble the collocation system stiffness matrix K in CSR format.

        Returns
        -------
        None.

        """
        # pre-allocate array for indices
        # this is the maximum possibly required size; not all will be used
        nMaxEntriesPerNode = int(self.nNodes*4*(self.support+0.25/self.N)**2)
        data = np.empty(self.nNodes * nMaxEntriesPerNode, dtype='float64')
        indices = np.empty(self.nNodes * nMaxEntriesPerNode, dtype='uint32')
        indptr = np.empty(self.nNodes+1, dtype='uint32')
        index = 0
        for iN, node in enumerate(self.nodes):
            inds = self.defineSupport(node)
            nEntries = len(inds)
            indptr[iN] = index
            indices[index:index+nEntries] = inds
            if (self.isBoundaryNode[iN]):
                phi = self.shapeFunctions0(node, inds)
                data[index:index+nEntries] = phi
            else:
                phi, d2phi = self.shapeFunctions2(node, inds)
                data[index:index+nEntries] = d2phi.sum(axis=1)
            index += nEntries
        indptr[-1] = index
        self.K = sp.csr_matrix( (data[0:index], indices[0:index], indptr),
                                shape=(self.nNodes, self.nNodes) )
    
    def solve(self, x0=None, tol=1e-05, maxiter=1000, M=None, callback=None,
              inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True,
              prepend_outer_v=False, atol=1e-05):
        """Solve for the approximate solution using an iterative solver.

        Returns
        -------
        None.

        """
        uTmp, self.info = sp_la.lgmres(self.K, self.b,
            x0, tol, maxiter, M, callback, inner_m, outer_k, outer_v,
            store_outer_Av, prepend_outer_v, atol)
        # uTmp = sp_la.spsolve(self.K, self.b) # direct solver for testing
        if (self.info != 0):
            print(f'solution failed with error code: {self.info}')
        # reconstruct final u vector from shape functions
        self.u = np.empty(self.nNodes, dtype='float64')
        for iN, node in enumerate(self.nodes):
            indices = self.defineSupport(node)
            phi = self.shapeFunctions0(node, indices)
            self.u[iN] = uTmp[indices]@phi
    
    def cond(self):
        """Computes the condition number of the stiffness mat+
        rix K.

        Returns
        -------
        c : float
            The condition number of the matrix.

        """
        c = sp_la.norm(self.K)*sp_la.norm(sp_la.inv(self.K.tocsc()))
        return c
