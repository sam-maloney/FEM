# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

class Mesh(object):
    """ Simple Mesh class for a regular triangular grid on a unit square
    """
    def generate_mesh(self):
        for j in range(self.N + 1):
            for i in range(self.N + 1):
                self.nodes[j*(self.N+1) + i, 0] = i/self.N
                self.nodes[j*(self.N+1) + i, 1] = j/self.N
                if (i == 0) or (j == 0) or (i == self.N) or (j == self.N):
                    self.is_boundary_node[j*(self.N+1) + i] = True
        
        for j in range(self.N):
            for i in range(self.N):
                self.elems[2*j*self.N + 2*i, 0] = (j  )*(self.N+1) + i
                self.elems[2*j*self.N + 2*i, 1] = (j  )*(self.N+1) + i+1
                self.elems[2*j*self.N + 2*i, 2] = (j+1)*(self.N+1) + i+1
                
                self.elems[2*j*self.N + 2*i+1, 0] = (j  )*(self.N+1) + i
                self.elems[2*j*self.N + 2*i+1, 1] = (j+1)*(self.N+1) + i+1
                self.elems[2*j*self.N + 2*i+1, 2] = (j+1)*(self.N+1) + i
    
    def __init__(self, N):
        self.N = N
        self.elems = np.empty((2*N*N      , 3), dtype='int32')
        self.nodes = np.empty(((N+1)*(N+1), 2), dtype='float64')
        self.is_boundary_node = np.full((N+1)*(N+1), False)
        self.generate_mesh()
    
    def __repr__(self):
        return self.__class__.__name__ + '({!r})'.format(self.N)

    def __str__(self):
        string = repr(self) + '\n\nNodes:\n#\tx\ty\n'
        for j in range(self.N + 1):
            for i in range(self.N + 1):
                n = j*(self.N+1) + i
                string += (str(n) + '\t'
                           + "{:.5f}".format(self.nodes[n, 0]) + '\t'
                           + "{:.5f}".format(self.nodes[n, 1]) + '\n')
        
        string += '\nElements:\n#\tn1\tn2\tn3\n'
        for j in range(self.N):
            for i in range(self.N):
                n = 2*j*self.N + 2*i
                string += (str(n) + '\t'
                           + str(self.elems[n, 0]) + '\t'
                           + str(self.elems[n, 1]) + '\t'
                           + str(self.elems[n, 2]) + '\n')
                
                n = 2*j*self.N + 2*i+1
                string += (str(n) + '\t'
                           + str(self.elems[n, 0]) + '\t'
                           + str(self.elems[n, 1]) + '\t'
                           + str(self.elems[n, 2]) + '\n')
        return string
    
    def __len__(self):
        return len(self.elems)

    def __getitem__(self, position):
        return self.elems[position]
    
##### End of Mesh class definition #####


##### Begin helper functions #####

def FEM_initialize(N):
    # this is just a helper function, so we want everything global
    global mesh, data, row_ind, col_ind, n_nodes, F, g
    
    # generate the mesh
    mesh = Mesh(N)
    
    # pre-allocate arrays for stiffness matrix triplets
    # these are the maximum possibly required sizes; not all will be used
    data = np.zeros(9*2*N*N, dtype='float64')
    row_ind = np.zeros(9*2*N*N, dtype='int32')
    col_ind = np.zeros(9*2*N*N, dtype='int32')
    
    # pre-allocate array for RHS vector and Dirichlet boundary function
    n_nodes = (N+1)*(N+1)
    F = np.zeros(n_nodes, dtype='float64')
    g = np.zeros(n_nodes, dtype='float64')
    
    # compute boundary function values at boundary nodes
    g[mesh.is_boundary_node] = \
        ( np.sin(k*np.pi*mesh.nodes[mesh.is_boundary_node,0])
        * np.sinh(k*np.pi*mesh.nodes[mesh.is_boundary_node,1]) )


def compute_error(u, iN):
    global mesh, E_inf, E_2, k
    
    u_exact = np.sin(k*np.pi*mesh.nodes[:,0])*np.sinh(k*np.pi*mesh.nodes[:,1])
    E_inf[iN] = np.linalg.norm(u - u_exact, np.inf)
    E_2[iN] = np.linalg.norm(u - u_exact)/N
    print('max error =', E_inf[iN])
    print('L2 error  =', E_2[iN])
    
    return u_exact
        
##### End helper functions #####
        

##### Start of main program #####
        
# wavenumber for boundary function u(x,1) = sinh(k*pi) = g(x,y)
k = 1

# define shape-function gradients on reference triangle
gradients = np.array([[-1, 1, 0], [-1, 0, 1]])

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
    FEM_initialize(N)
    
    # triplet index for assembly
    index = 0
    
    for elem in mesh:
        N_a = mesh.nodes[elem[0]]
        N_b = mesh.nodes[elem[1]]
        N_c = mesh.nodes[elem[2]]
        # define transformation matrix for elem to reference triangle
        J = np.array([N_b - N_a, N_c - N_a]).transpose()
        det_J = np.linalg.det(J)
        J_inv_T = np.linalg.inv(J).transpose()
        
        for i_alpha, alpha in enumerate(elem):
            if mesh.is_boundary_node[alpha]:
                F[alpha] += g[alpha]
                for beta in elem:
                    if (beta == alpha):
                        data[index] = 1
                        row_ind[index] = alpha
                        col_ind[index] = beta
                        index += 1;
                        
            # if alpha is not a boundary node
            else:
                for i_beta, beta in enumerate(elem):
                    if mesh.is_boundary_node[beta]:
                        F[alpha] -= ( g[beta] * det_J
                                    * np.dot(J_inv_T@gradients[:,i_alpha],
                                             J_inv_T@gradients[:,i_beta]) )
                    else:
                        data[index] = det_J*np.dot(J_inv_T@gradients[:,i_alpha],
                                                   J_inv_T@gradients[:,i_beta])
                        row_ind[index] = alpha
                        col_ind[index] = beta
                        index += 1;
    
    # assemble the triplets into the sparse stiffness matrix
    A = csr_matrix((data, (row_ind, col_ind)), shape=(n_nodes, n_nodes))
    
    # solve for the approximate solution
    u, info = cg(A, F)
#    if (info == 0):
#        print('Solution successful')
    if (info != 0):
        print('solution failed for N = ', N, ' with error code:', info)
    
    # compute the analytic solution and error norms
    u_exact = compute_error(u, iN)
    
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
triangulation = mpl.tri.Triangulation(mesh.nodes[:,0],
                                      mesh.nodes[:,1],
                                      mesh.elems)

# plot the result
plt.subplot(221)
plt.tripcolor(triangulation, u, shading='gouraud')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Final FEM solution')
## plot the mesh over the FEM solution
#plt.triplot(triangulation, 'w-', lw=1)

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
plt.yticks([1, 1.5, 2, 2.5, 3])
plt.xlabel(r'$N$')
plt.ylabel(r'Intra-step Order of Convergence')
plt.title('FEM Order of Accuracy')
plt.legend(fontsize='x-large')
