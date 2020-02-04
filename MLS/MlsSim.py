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
    """Class for meshless moving least squares (MLS) method.
    
    Parameters
    ----------
    N : integer
        Number of grid cells along one dimension.
        Must be greater than 0.
    g : function object
        Function defining the solution Dirichlet values along the boundary.
        The object must take an nx2 numpy.ndarray of points and return a
        1D numpy.ndarray of size n for the function values at those points.
    Nquad : integer, optional
        Number of quadrature points in each grid cell along one dimension.
        Must be either 1 or 2.
        The default is 1.
    support : float, optional
        The size of the shape function support, given as a multiple of the
        grid spacing for the given N if the value is positive.
        Supplying a negative value leads to default support sizes being used,
        namely 1.4 for the Galerkin method or 2.6 for the collocation method.
        The default is -1.
    form : string, optional
        Form of the spline used for the kernel weighting function.
        Must be either 'cubic' or 'quartic'.
        The default is 'cubic'.
    method : string, optional
        Method used for assembling the stiffness matrix.
        Must be either 'galerkin' or 'collocation'.
        The default is 'galerkin'.
    quadrature : string, optional
        Distribution of quadrature points in each cell.
        Must be either 'uniform' or 'gaussian'.
        The default is 'uniform'.
    """
    
    def __init__(self, N, g, Nquad=1, support=-1, form='cubic',
                 method='galerkin', quadrature='uniform'):
        self.N = N
        self.nCells = N*N
        self.nNodes = (N+1)*(N+1)
        self.Nquad = Nquad
        if support > 0:
            self.support = support/N
        else: # if support is negative, set to default grid spacing
            if method == 'galerkin':
                self.support = 1.4/N
            elif method == 'collocation':
                self.support = 2.6/N
            else: # if method is unkown
                self.support = 1.4/N
        self.nodes = ( np.indices((N+1, N+1), dtype='float64')
                       .T.reshape(-1,2) ) / N
        self.isBoundaryNode = np.any(np.mod(self.nodes, 1) == 0, axis=1)
        self.nBoundaryNodes = np.count_nonzero(self.isBoundaryNode)
        self.boundaryValues = g(self.nodes[self.isBoundaryNode]) \
                               .round(decimals=14)
        self.g = g
        self.selectSpline(form)
        self.selectMethod(method, quadrature)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.N},{self.g}," \
               f"{self.Nquad},{self.support},'{self.form}','{self.method}')"
    
    def selectMethod(self, method, quadrature):
        """Register the 'self.assembleStiffnesMatrix' method.
        
        Parameters
        ----------
        method : string
            Method used for assembling the stiffness matrix.
            Must be either 'galerkin' or 'collocation'.
        quadrature : string
            Distribution of quadrature points in each cell.
            Must be either 'uniform' or 'gaussian'.

        Returns
        -------
        None.

        """
        if method == 'galerkin':
            self.assembleStiffnessMatrix = self.assembleGalerkinStiffnessMatrix
            self.method = method
            self.generateQuadraturePoints(quadrature)
            self.b = np.concatenate((np.zeros(self.nNodes,dtype='float64'),
                                     self.boundaryValues))
        elif method == 'collocation':
            self.assembleStiffnessMatrix = self.assembleCollocationStiffnessMatrix
            self.method = method
            self.b = np.zeros(self.nNodes,dtype='float64')
            self.b[self.isBoundaryNode] = self.boundaryValues
        else:
            print(f"Error: unkown assembly method '{method}'. "
                  f"Must be one of 'galerkin' or 'collocation'.")
    
    def generateQuadraturePoints(self, quadrature):
        """Compute array of quadrature points for Galerkin integration.

        Parameters
        ----------
        quadrature : string
            Distribution of quadrature points in each cell.
            Must be either 'uniform' or 'gaussian'.

        Returns
        -------
        None.

        """
        self.quadrature = quadrature
        if self.Nquad == 1:
            self.quads = ( np.indices((self.N, self.N), dtype='float64')
                           .T.reshape(-1,2) + 0.5 ) / self.N
            self.quadWeight = 1.0/(self.N*self.N)
            self.nQuads = len(self.quads)
        elif self.Nquad == 2:
            tmp = ( np.indices((self.N, self.N), dtype='float64')
                           .T.reshape(-1,2) + 0.5 ) / self.N
            if quadrature == 'uniform':
                offset = 0.25/self.N
            elif quadrature == 'gaussian':
                offset = 0.5/(np.sqrt(3.0)*self.N)
            self.quads = np.concatenate((
                tmp - offset,
                tmp + offset,
                np.hstack((tmp[:,0:1] + offset, tmp[:,1:2] - offset)),
                np.hstack((tmp[:,0:1] - offset, tmp[:,1:2] + offset)) ))
            self.quadWeight = 0.25/(self.N*self.N)
            self.nQuads = len(self.quads)
        else:
            print(f"Error: bad Nquad value of '{self.Nquad}'. "
                  f"Must be either 1 or 2.")
    
    def selectSpline(self, form):
        """Register the 'self.spline' method to the correct order kernel.
        
        Parameters
        ----------
        form : string
            Form of the spline used for the kernel weighting function.
            Must be either 'cubic' or 'quartic'.

        Returns
        -------
        None.

        """
        self.form = form
        if form == 'cubic':
            self.spline0 = self.cubicSpline0
            self.spline1 = self.cubicSpline1
            self.spline2 = self.cubicSpline2
        elif form == 'quartic':
            self.spline0 = self.quarticSpline0
            self.spline1 = self.quarticSpline1
            self.spline2 = self.quarticSpline2
        else:
            print(f"Error: unkown spline form '{form}'. "
                  f"Must be one of 'cubic' or 'quartic'.")
    
    def cubicSpline0(self, r):
        """Compute cubic spline function value.

        Parameters
        ----------
        r : numpy.array([...], dtype='float64')
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.array([...], dtype='float64')
            Values of the cubic spline function at the given distances.

        """
        i0 = r < 0.5
        i1 = np.logical_xor(r < 1, i0)
        w = np.zeros(r.size, dtype='float64')
        if i0.any():
            r1 = r[i0]
            r2 = r1*r1
            r3 = r2*r1
            w[i0] = 2.0/3.0 - 4.0*r2 + 4.0*r3
        if i1.any():
            r1 = r[i1]
            r2 = r1*r1
            r3 = r2*r1
            w[i1] = 4.0/3.0 - 4.0*r1 + 4.0*r2 - 4.0/3.0*r3
        return w
    
    def cubicSpline1(self, r):
        """Compute cubic spline function and its radial derivative.

        Parameters
        ----------
        r : numpy.array([...], dtype='float64')
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.array([...], dtype='float64')
            Values of the cubic spline function at the given distances.
        dwdr : numpy.array([...], dtype='float64')
            Values of the radial derivative at the given distances.

        """
        i0 = r < 0.5
        i1 = np.logical_xor(r < 1, i0)
        w = np.zeros(r.size, dtype='float64')
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]
            r2 = r1*r1
            r3 = r2*r1
            w[i0] = 2.0/3.0 - 4.0*r2 + 4.0*r3
            dwdr[i0] = -8.0*r1 + 12.0*r2
        if i1.any():
            r1 = r[i1]
            r2 = r1*r1
            r3 = r2*r1
            w[i1] = 4.0/3.0 - 4.0*r1 + 4.0*r2 - 4.0/3.0*r3
            dwdr[i1] = -4.0 + 8.0*r1 - 4.0*r2
        return w, dwdr
    
    def cubicSpline2(self, r):
        """Compute cubic spline function and its radial derivatives.

        Parameters
        ----------
        r : numpy.array([...], dtype='float64')
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.array([...], dtype='float64')
            Values of the cubic spline function at the given distances.
        dwdr : numpy.array([...], dtype='float64')
            Values of the radial derivative at the given distances.
        d2wdr2 : numpy.array([...], dtype='float64')
            Values of the 2nd order radial derivative at the given distances.

        """
        i0 = r < 0.5
        i1 = np.logical_xor(r < 1, i0)
        w = np.zeros(r.size, dtype='float64')
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]
            r2 = r1*r1
            r3 = r2*r1
            w[i0] = 2.0/3.0 - 4.0*r2 + 4.0*r3
            dwdr[i0] = -8.0*r1 + 12.0*r2
            d2wdr2[i0] = -8.0 + 24.0*r1
        if i1.any():
            r1 = r[i1]
            r2 = r1*r1
            r3 = r2*r1
            w[i1] = 4.0/3.0 - 4.0*r1 + 4.0*r2 - 4.0/3.0*r3
            dwdr[i1] = -4.0 + 8.0*r1 - 4.0*r2
            d2wdr2[i1] = 8.0 - 8.0*r1
        return w, dwdr, d2wdr2
    
    def quarticSpline0(self, r):
        """Compute quartic spline function values.

        Parameters
        ----------
        r : numpy.array([...], dtype='float64')
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.array([...], dtype='float64')
            Values of the quartic spline function at the given distances.

        """
        i0 = r < 1
        w = np.zeros(r.size, dtype='float64')
        if i0.any():
            r1 = r[i0]
            r2 = r1*r1
            r3 = r2*r1
            r4 = r2*r2
            w[i0] = 1.0 - 6.0*r2 + 8.0*r3 - 3.0*r4
        return w
    
    def quarticSpline1(self, r):
        """Compute quartic spline function and radial derivative values.

        Parameters
        ----------
        r : numpy.array([...], dtype='float64')
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.array([...], dtype='float64')
            Values of the quartic spline function at the given distances.
        dwdr : numpy.array([...], dtype='float64')
            Values of the radial derivative at the given distances.

        """
        i0 = r < 1
        w = np.zeros(r.size, dtype='float64')
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]
            r2 = r1*r1
            r3 = r2*r1
            r4 = r2*r2
            w[i0] = 1.0 - 6.0*r2 + 8.0*r3 - 3.0*r4
            dwdr[i0] = -12.0*r1 + 24.0*r2 - 12.0*r3
        return w, dwdr
    
    def quarticSpline2(self, r):
        """Compute quartic spline function and radial derivative values.

        Parameters
        ----------
        r : numpy.array([...], dtype='float64')
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.array([...], dtype='float64')
            Values of the quartic spline function at the given distances.
        dwdr : numpy.array([...], dtype='float64')
            Values of the radial derivative at the given distances.
        d2wdr2 : numpy.array([...], dtype='float64')
            Values of the 2nd order radial derivative at the given distances.

        """
        i0 = r < 1
        w = np.zeros(r.size, dtype='float64')
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]
            r2 = r1*r1
            r3 = r2*r1
            r4 = r2*r2
            w[i0] = 1.0 - 6.0*r2 + 8.0*r3 - 3.0*r4
            dwdr[i0] = -12.0*r1 + 24.0*r2 - 12.0*r3
            d2wdr2[i0] = -12.0 + 48.0*r1 - 36.0*r2
        return w, dwdr, d2wdr2
    
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
        distances = la.norm(point - self.nodes[indices], axis=-1)/self.support
        w = self.spline0(distances)
        p = np.hstack((np.ones((len(indices),1)), self.nodes[indices]))
        A = w*p.T@p
        # --------------------------------------
        #      compute  matrix c(x) and phi
        # --------------------------------------
        # A(x)c(x) = p(x)
        # Backward substitution for c(x) using LU factorization for A
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
        gradphi : nx2 numpy.ndarray, dtype='float64'
            Gradients of phi for all nodes in indices. [[dx1,dy1],[dx2,dy2]...]

        """
        # --------------------------------------
        #     compute the moment matrix A(x)
        # --------------------------------------
        displacement = (point - self.nodes[indices])/self.support
        distance = np.array(la.norm(displacement, axis=-1))
        w, dwdr = self.spline1(distance)
        i0 = distance > 1e-14
        gradr = np.full(self.nodes[indices].shape, np.sqrt(0.5)/self.support, dtype='float64')
        gradr[i0] = displacement[i0] / \
                    (distance[i0]*self.support).reshape((-1,1))
        gradw = dwdr.reshape((-1,1)) * gradr
        p = np.hstack((np.ones((len(indices),1)), self.nodes[indices]))
        A = w*p.T@p
        dAdx = gradw[:,0]*p.T@p
        dAdy = gradw[:,1]*p.T@p
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
        grad2phi : nx2 numpy.ndarray, dtype='float64'
            2nd derivatives of phi for all nodes in indices.
            [[dxx1,dyy1],[dxx2,dyy2]...]

        """
        # --------------------------------------
        #     compute the moment matrix A(x)
        # --------------------------------------
        displacement = (point - self.nodes[indices])/self.support
        distance = np.array(la.norm(displacement, axis=-1))
        w, dwdr, d2wdr2 = self.spline2(distance)
        i0 = distance > 1e-14
        gradr = np.full(self.nodes[indices].shape, np.sqrt(0.5)/self.support, dtype='float64')
        gradr[i0] = displacement[i0] / \
                    (distance[i0]*self.support).reshape((-1,1))
        gradw = dwdr.reshape((-1,1)) * gradr
        grad2w = d2wdr2.reshape((-1,1)) * gradr*gradr
        p = np.hstack((np.ones((len(indices),1)), self.nodes[indices]))
        A = w*p.T@p
        dAdx = gradw[:,0]*p.T@p
        dAdy = gradw[:,1]*p.T@p
        d2Adx2 = grad2w[:,0]*p.T@p
        d2Ady2 = grad2w[:,1]*p.T@p
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
        ##### Alternative assembly method, turned out to be slower #####
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
        """Computes the condition number of the stiffness matrix K.

        Returns
        -------
        c : float
            The condition number of the matrix.

        """
        c = sp_la.norm(self.K)*sp_la.norm(sp_la.inv(self.K.tocsc()))
        return c
