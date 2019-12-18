# -*- coding: utf-8 -*-
"""
Simple linear finite element method (FEM) simulation

@author: Sam Maloney
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

class Mesh(object):
    """ Simple Mesh class for a regular triangular grid on a unit square."""        
    
    def __init__(self, N):
        self.N = N
        self.nodes = np.indices((N+1, N+1)).T.reshape(-1,2) / N
        self.is_boundary_node = np.any(np.mod(self.nodes, 1) == 0, axis=1)
        self.elems = np.empty((2*N*N, 3), dtype='int32')
        self.elems[0:2*N,:] = np.array([[range(N),range(N)],
                                        [range(1,N+1),range(N+2,2*N+2)],
                                        [range(N+2,2*N+2),range(N+1,2*N+1)]]) \
                                        .T.reshape(-1,3)
        for i in range(1,N):
            self.elems[2*i*N:2*(i+1)*N,:] = self.elems[0:2*N,:] + i*(N+1)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.N})'

    def __str__(self):
        string = f'{repr(self)}\n\nNodes:\n#\tx\ty\n'
        for n, (x, y) in enumerate(self.nodes):
            string += f'{n}\t{x:.5f}\t{y:.5f}\n'
        string += '\nElements:\n#\tn1\tn2\tn3\n'
        for e, (a, b, c) in enumerate(self.elems):
            string += f'{e}\t{a}\t{b}\t{c}\n'
        return string
    
    def __len__(self):
        return len(self.elems)

    def __getitem__(self, position):
        return self.elems[position]
    
##### End of Mesh class definition #####

class FemSim(object):
    """Linear FEM simulation class."""
    
    def __init__(self, N, k=1):
        self.N = N
        self.k = k
        self.mesh = Mesh(N)
        
        # pre-allocate arrays for stiffness matrix triplets
        # these are the maximum possibly required sizes; not all will be used
        self.data = np.zeros(9*2*N*N, dtype='float64')
        self.row_ind = np.zeros(9*2*N*N, dtype='int32')
        self.col_ind = np.zeros(9*2*N*N, dtype='int32')
        
        # pre-allocate array for RHS vector and Dirichlet boundary function
        self.n_nodes = (N+1)*(N+1)
        self.F = np.zeros(self.n_nodes, dtype='float64')
        self.g = np.zeros(self.n_nodes, dtype='float64')
        
        # compute boundary function values at boundary nodes
        self.g[self.mesh.is_boundary_node] = (
              np.sin(k*np.pi*self.mesh.nodes[self.mesh.is_boundary_node,0])
            * np.sinh(k*np.pi*self.mesh.nodes[self.mesh.is_boundary_node,1]) )
    
    def assembleStiffnessMatrix(self):
        # define shape-function gradients on reference triangle
        gradients = np.array([[-1, 1, 0], [-1, 0, 1]])
        
        # triplet index for assembly
        index = 0
        
        for elem in self.mesh:
            N_a = self.mesh.nodes[elem[0]]
            N_b = self.mesh.nodes[elem[1]]
            N_c = self.mesh.nodes[elem[2]]
            # define transformation matrix for elem to reference triangle
            J = np.array([N_b - N_a, N_c - N_a]).transpose()
            det_J = np.linalg.det(J)
            J_inv_T = np.linalg.inv(J).transpose()
            
            for i_alpha, alpha in enumerate(elem):
                if self.mesh.is_boundary_node[alpha]:
                    self.F[alpha] += self.g[alpha]
                    for beta in elem:
                        if (beta == alpha):
                            self.data[index] = 1
                            self.row_ind[index] = alpha
                            self.col_ind[index] = beta
                            index += 1;
                    continue # to next alpha iteration
                # if alpha is not a boundary node
                for i_beta, beta in enumerate(elem):
                    if self.mesh.is_boundary_node[beta]:
                        self.F[alpha] -= ( self.g[beta] * det_J
                                    * np.dot(J_inv_T@gradients[:,i_alpha],
                                             J_inv_T@gradients[:,i_beta]) )
                        continue # to next beta iteration
                    # if beta is not a boundary node
                    self.data[index] = det_J*np.dot(J_inv_T@gradients[:,i_alpha],
                                               J_inv_T@gradients[:,i_beta])
                    self.row_ind[index] = alpha
                    self.col_ind[index] = beta
                    index += 1;
        
        # assemble the triplets into the sparse stiffness matrix
        self.A = csr_matrix((self.data, (self.row_ind, self.col_ind)),
                            shape=(self.n_nodes, self.n_nodes))
    
    def solve(self):
        """Solve for the approximate solution."""
        self.u, self.info = cg(self.A, self.F)
        
    def __repr__(self):
        return f'{self.__class__.__name__}({self.N},{self.k})'

##### End of FemSim class definition #####
        

##### Start of main program #####
        
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
    femSim = FemSim(N, k)
    
    # Assemble the stiffness matrix and solve for the approximate solution
    femSim.assembleStiffnessMatrix()
    femSim.solve();
    
#    if (femSim.info == 0):
#        print('Solution successful')
    if (femSim.info != 0):
        print(f'solution failed for N = {N} with error code: {femSim.info}')
    
    # compute the analytic solution and error norms
    u_exact = ( np.sin(femSim.k*np.pi*femSim.mesh.nodes[:,0])
               *np.sinh(femSim.k*np.pi*femSim.mesh.nodes[:,1]) )
    E_inf[iN] = np.linalg.norm(femSim.u - u_exact, np.inf)
    E_2[iN] = np.linalg.norm(femSim.u - u_exact)/N
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

# create a matplotlib triangulation object from the mesh
triangulation = mpl.tri.Triangulation(femSim.mesh.nodes[:,0],
                                      femSim.mesh.nodes[:,1],
                                      femSim.mesh.elems)

# plot the result
plt.subplot(221)
plt.tripcolor(triangulation, femSim.u, shading='gouraud')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Final FEM solution')

# plot analytic solution
plt.subplot(222)
plt.tripcolor(triangulation, u_exact, shading='gouraud')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Analytic solution')

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

## plot the mesh over the FEM solution
#fig = plt.figure()
#fig.set_size_inches(7,7)
#plt.tripcolor(triangulation, u, shading='gouraud')
#plt.triplot(triangulation, 'r-', lw=2)
#plt.xlabel(r'$x$')
#plt.ylabel(r'$y$')
#plt.axes().set_aspect('equal')
#plt.margins(0,0)
#plt.savefig("FEM_mesh.pdf", bbox_inches = 'tight',
#    pad_inches = 0)