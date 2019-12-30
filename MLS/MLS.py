#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Meshfree method simulation using moving least squares (MLS)

@author: Sam Maloney
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse as sp

class MlsSim(object):
    """Class for meshless moving least squares (MLS) method."""
    
    def __init__(self, N, k=1, Nquad=1, support=-1, form='cubic'):
        self.N = N
        self.k = k
        self.nCells = N*N
        self.nNodes = (N+1)*(N+1)
        self.Nquad = Nquad
        if support > 0:
            self.support = support
        else: # if support is negative, set to default 1.5X grid spacing
            self.support = 1.6/(N)
        self.selectSpline(form)
        self.nodes = ( np.indices((N+1, N+1), dtype='float64')
                       .T.reshape(-1,2) ) / N
        self.isBoundaryNode = np.any(np.mod(self.nodes, 1) == 0, axis=1)
        self.nBoundaryNodes = np.count_nonzero(self.isBoundaryNode)
        self.generateQuadraturePoints(N, Nquad)
        self.g = ( np.sin (k*np.pi*self.nodes[self.isBoundaryNode,0])
                 * np.sinh(k*np.pi*self.nodes[self.isBoundaryNode,1]) ).round(
                   decimals=14)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.N},{self.k}," \
               f"{self.Nquad},{self.support},'{self.form}')"
    
    def generateQuadraturePoints(self, N, Nquad):
        self.quads = ( np.indices((N, N), dtype='float64')
                       .T.reshape(-1,2) + 0.5 ) / N
        self.quadWeight = 1.0/(N*N)
        self.nQuads = len(self.quads)
    
    def cubicSpline(self, r):
        """Compute cubic spline function and its radial derivative.

        Parameters
        ----------
        r : float
            Distance from quadrature point to node point.

        Returns
        -------
        w : float
            Value of the cubic spline function at the given distance.
        dwdr : float
            Value of the radial derivative at the given distance.

        """
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
        """Compute quartic spline function and its radial derivative.

        Parameters
        ----------
        r : float
            Distance from quadrature point to node point.

        Returns
        -------
        w : float
            Value of the quartic spline function at the given distance.
        dwdr : float
            Value of the radial derivative at the given distance.

        """
        if r < 1.0:
            w = 1.0 - 6.0*r*r + 8.0*r*r*r - 3.0*r*r*r*r
            dwdr = -12.0*r + 24.0*r*r - 12.0*r*r*r
        else:
            w = 0.0
            dwdr = 0.0
        return w, dwdr
    
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
    
    def weightFunction(self, quadPoint, nodePoint):
        """Compute weight function and gradient using cubic or quartic spline.        

        Parameters
        ----------
        quadPoint : numpy.array([x,y], dtype='float64')
            Coordinates of quadrature point at which w is evaluated.
        nodePoint : numpy.array([x,y], dtype='float64')
            Coordinates of node I.

        Returns
        -------
        w : float
            Value of the weight function at the quadrature point.
        gradw : numpy.array([dx,dy], dtype='float64')
            Gradient of weight function at the quadrature point.

        """
        displacement = (quadPoint - nodePoint)/self.support
        distance = la.norm(displacement)
        w, dwdr = self.spline(distance)
        if distance > 1e-14:
            gradr = displacement/(distance*self.support)
            gradw = dwdr * gradr
        else:
            gradw = np.zeros(2,dtype='float64')
        return w, gradw

    def shapeFunctions(self, quadPoint, indices, check = False):
        """Compute shape function at quad point for all nodes in its support.
        Basis used is linear basis pT = [1 x y].

        Parameters
        ----------
        quadPoint : numpy.array([x,y], dtype='float64')
            Coordinate of quadrature point.
        indices : numpy.array([...], dtype='uint32')
            Indices of nodes within support of given quadrature point.
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
        gradw = np.empty((len(indices), 2), dtype='float64')
        for i, node in enumerate(self.nodes[indices]):
            wi, gradwi = self.weightFunction(quadPoint, node)
            pTp = np.outer([1.0, node[0], node[1]], [1.0, node[0], node[1]])
            A = A + wi*pTp
            dAdx = dAdx + gradwi[0]*pTp
            dAdy = dAdy + gradwi[1]*pTp
            w[i] = wi
            gradw[i,:] = gradwi
        # --------------------------------------
        #         compute  matrix c(x)
        # --------------------------------------
        # A(x)c(x)   = p(x)
        # A(x)c_k(x) = b_k(x)
        # Backward substitutions, once for c(x), twice for c_k(x) k=1,2
        # Using LU factorization for A
        p = np.concatenate(([1.0], quadPoint))
        lu, piv = la.lu_factor(A, check_finite=check)
        c = np.empty((3,3), dtype='float64')
        c[0] = la.lu_solve((lu, piv), p, check_finite=check)
        c[1] = la.lu_solve((lu, piv),([0,1,0] - dAdx@c[0]), check_finite=check)
        c[2] = la.lu_solve((lu, piv),([0,0,1] - dAdy@c[0]), check_finite=check)
        # --------------------------------------
        #       compute phi and gradphi
        # --------------------------------------
        phi = np.empty((len(indices)), dtype='float64')
        gradphi = np.empty((len(indices), 2), dtype='float64')
        for i, node in enumerate(self.nodes[indices]):
            pi = np.concatenate(([1.0], node))
            cpi = c[0] @ pi
            phi[i] = cpi * w[i]
            gradphi[i,:] = (c[1:3,:] @ pi * w[i]).T + cpi*gradw[i]
        return phi, gradphi
        
    def defineSupport(self, quadPoint):
        """Find nodes within support of a given quadrature point.

        Parameters
        ----------
        quadPoint : numpy.array([x,y], dtype='float64')
            Coordinates of given quadrature point.

        Returns
        -------
        indices : numpy.array([...], dtype='uint32')
            Indices of nodes within support of given quadrature point.
            
        """
        distances = la.norm(quadPoint - self.nodes, axis=1)
        indices = np.flatnonzero(distances < self.support).astype('uint32')
        return indices
    
    def assembleStiffnessMatrix(self):
        """Assemble the system stiffness matrix K in CSR format.

        Returns
        -------
        None.
        
        # K : scipy.sparse.csr_matrix
        #     Galerkin stiffness matrix for the MLS system.

        """
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
            phi, gradphi = self.shapeFunctions(quad, indices)
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
            phi, gradphi = self.shapeFunctions(node, indices)
            data[index:index+nEntries] = -1.0*phi
            row_ind[index:index+nEntries] = indices
            col_ind[index:index+nEntries] = np.repeat(iN, nEntries)
            index += nEntries
        inds = np.flatnonzero(data.round(decimals=14,out=data))
        G = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
                           shape=(self.nNodes, self.nBoundaryNodes) )
        G *= -1.0
        self.K = sp.bmat([[self.K, G], [G.T, None]], format='csr')
        # return self.K
    
    def solve(self, x0=None, tol=1e-05, maxiter=1000, M=None, callback=None,
              inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True,
              prepend_outer_v=False, atol=1e-05):
        """Solve for the approximate solution.

        Returns
        -------
        None.

        """
        self.b = np.concatenate((np.zeros(self.nNodes,dtype='float64'),
                                 self.g))
        uTmp, self.info = sp.linalg.lgmres(self.K, self.b,
            x0, tol, maxiter, M, callback, inner_m, outer_k, outer_v,
            store_outer_Av, prepend_outer_v, atol)
        # uTmp = sp.linalg.spsolve(self.K, self.b) # direct solver for testing
        if (self.info != 0):
            print(f'solution failed with error code: {self.info}')
        # reconstruct final u vector from shape functions
        self.u = np.empty(self.nNodes, dtype='float64')
        for iN, node in enumerate(self.nodes):
            indices = self.defineSupport(node)
            phi, gradphi = self.shapeFunctions(node, indices)
            self.u[iN] = uTmp[indices]@phi

##### End of MlsSim class definition #####
            

##### Start of main program #####
            
# mls = MlsSim(10)
# mls.assembleStiffnessMatrix()
        
# wavenumber for boundary function u(x,1) = sinh(k*pi) = g(x,y)
k = 1

# allocate arrays for convergence testing
start = 1
stop = 6
nSamples = stop - start + 1
N_array = np.logspace(start, stop, num=nSamples, base=2, dtype='int32')
E_inf = np.empty(nSamples, dtype='float64')
E_2 = np.empty(nSamples, dtype='float64')

# loop over N to test convergence where N is the number of
# grid cells along one dimension, each cell forms 2 triangles
# therefore number of triangular elements equals 2*N*N
# and number of nodes equals (N+1)*(N+1)
for iN, N in enumerate(N_array):

    print('N =', N)
    
    # allocate arrays and compute boundary values
    mlsSim = MlsSim(N, k)
    
    # Assemble the stiffness matrix and solve for the approximate solution
    tolerance = 1e-10
    mlsSim.assembleStiffnessMatrix()
    mlsSim.solve(tol=tolerance, atol=tolerance)
    
    # compute the analytic solution and error norms
    u_exact = ( np.sin(mlsSim.k*np.pi*mlsSim.nodes[:,0])
                *np.sinh(mlsSim.k*np.pi*mlsSim.nodes[:,1]) )
    E_inf[iN] = np.linalg.norm(mlsSim.u - u_exact, np.inf)
    E_2[iN] = np.linalg.norm(mlsSim.u - u_exact)/N
    print('max error =', E_inf[iN])
    print('L2 error  =', E_2[iN])
    
##### End of loop over N #####
    
    
##### Begin Plotting Routines #####

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(15,15)
mpl.rc('axes', titlesize='xx-large', labelsize='x-large')
mpl.rc('xtick', labelsize='large')
mpl.rc('ytick', labelsize='large')
plt.subplots_adjust(hspace = 0.3, wspace = 0.25)

# plot the error convergence
plt.subplot(223)
plt.loglog(N_array, E_inf, '.-', label=r'$E_\inf$')
plt.loglog(N_array, E_2, '.-', label=r'$E_2$')
plt.minorticks_off()
plt.xticks(N_array, N_array)
plt.xlabel(r'$N$')
plt.ylabel(r'Magnitude of Error Norm')
plt.title('FEM Error Norms')
plt.legend(fontsize='x-large')
## add labels for order of convergence to the same plot
#first = 1 # exclude 0th sample, as it is visually non-asymptotic
#p_inf = np.polyfit(np.log10(N_array[first:]), np.log10(E_inf[first:]), 1)
#p_2 = np.polyfit(np.log10(N_array[first:]), np.log10(E_2[first:]), 1)
#plt.text(N_array[3], E_inf[3],
#         '    $O(N^{' + "{:.2f}".format(p_inf[0]) + '})$',
#         fontsize='large')
#plt.text(N_array[3], E_2[3],
#         r'$O\left(N^{' + "{:.2f}".format(p_2[0]) + r'}\right)$    ',
#         fontsize='large', horizontalalignment='right',
#         verticalalignment='top')

# plot the intra-step order of convergence
plt.subplot(224)
logN = np.log(N_array)
logE_inf = np.log(E_inf)
logE_2 = np.log(E_2)
order_inf = (logE_inf[0:-1] - logE_inf[1:])/(logN[1:] - logN[0:-1])
order_2 = (logE_2[0:-1] - logE_2[1:])/(logN[1:] - logN[0:-1])
intraN = np.logspace(start+0.5, stop-0.5, num=nSamples-1, base=2.0)
plt.plot(intraN, order_inf, '.-', label=r'$E_\inf$')
plt.plot(intraN, order_2, '.-', label=r'$E_2$')
plt.plot([N_array[0], N_array[-1]], [2, 2], 'k:', label='Expected')
plt.xlim(N_array[0], N_array[-1])
plt.xscale('log')
plt.minorticks_off()
plt.xticks(N_array, N_array)
plt.ylim(1, 3)
plt.yticks([1, 1.5, 2, 2.5, 3])
plt.xlabel(r'$N$')
plt.ylabel(r'Intra-step Order of Convergence')
plt.title('FEM Order of Accuracy')
plt.legend(fontsize='x-large')