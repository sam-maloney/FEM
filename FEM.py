# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv, det
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

from math import sqrt, sin, sinh, pi

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
    
# End of Mesh class definition
        

##### Start of main program #####
        
# wavenumber for boundary function u(x,1) = sinh(k*pi) = g(x,y)
k = 1

# Numer of grid divisions along one dimension
# Number of triangular elements equals 2*N*N
N = 2
mesh = Mesh(N)

# pre-allocate arrays for stiffness matrix triplets
data = np.empty(9*2*N*N, dtype='float64')
row_ind = np.empty(9*2*N*N, dtype='int32')
col_ind = np.empty(9*2*N*N, dtype='int32')

# pre-allocate array for RHS vector and Dirichlet boundary function
n_nodes = (N+1)*(N+1)
F = np.zeros(n_nodes, dtype='float64')
g = np.zeros(n_nodes, dtype='float64')

# compute boundary function values at boundary nodes
#is_top_boundary = ( (mesh.nodes[:,1] == 1.0)
#                * (mesh.nodes[:,0] != 0.0)
#                * (mesh.nodes[:,0] != 1.0) )
#g[is_top_boundary] = sinh(k*pi)

for i_node, node in enumerate(mesh.nodes):
    if ( node[1] == 1.0 ):
        g[i_node] = sin(k*pi*node[0]) * sinh(k*pi)

# define shape-function gradients on reference triangle
gradients = np.array([[-sqrt(2), -1, 0], [-sqrt(2), 0, -1]])
#grad_a = np.array([[-sqrt(2)], [-sqrt(2)]])
#grad_b = np.array([[-1], [0]])
#grad_c = np.array([[0], [-1]])

# triplet index for assembly
index = 0

for elem in mesh:
    N_a = mesh.nodes[elem[0]]
    N_b = mesh.nodes[elem[1]]
    N_c = mesh.nodes[elem[2]]
    # define transformation matrix for elem to reference triangle
    J = np.array([N_b - N_a, N_c - N_a]).transpose()
    det_J = det(J)
    J_inv_T = inv(J).transpose()
    
    for i_alpha, alpha in enumerate(elem):
        if mesh.is_boundary_node[alpha]:
            # if alpha is on the top boundary, where y = 1
            if ( (mesh.nodes[alpha,1] == 1.0)
                    and not (mesh.nodes[alpha,0] == 0.0)
                    and not (mesh.nodes[alpha,0] == 1.0) ):
                F[alpha] += g[alpha]
                # for homogenous case all interior values of F are zero
                # so this is the only place we add to F
            for beta in elem:
                if (beta == alpha):
                    data[index] = 1
                    row_ind[index] = alpha
                    col_ind[index] = beta
                    index += 1;
                else:
                    data[index] = 0
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
                    data[index] = 0
                    row_ind[index] = alpha
                    col_ind[index] = beta
                    index += 1;
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
if (info == 0):
    print('Solution successful')
else:
    print('solution failed with error code:', info)

# compute the analytic solution
u_exact = np.empty(n_nodes, dtype='float64')
for i_node, node in enumerate(mesh.nodes):
    u_exact[i_node] = sin(k*pi*node[0])*sinh(k*pi*node[1])

E_inf = abs(u - u_exact).max()
print('max error = ', E_inf)

triangulation = mpl.tri.Triangulation(mesh.nodes[:,0],
                                      mesh.nodes[:,1],
                                      mesh.elems)

# plot the mesh
#plt.triplot(triangulation)

# plot the result
plt.subplot(211)
plt.tripcolor(triangulation, u, shading='gouraud')
plt.colorbar()

# plot analytic solution
plt.subplot(212)
plt.tripcolor(triangulation, u_exact, shading='gouraud')
plt.colorbar()

B = A.toarray()