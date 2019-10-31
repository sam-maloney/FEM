# -*- coding: utf-8 -*-

import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt

# Simple Mesh class for a regular triangular grid on a unit square
class Mesh(object):
    def generate_mesh(self):
        for j in range(self.N + 1):
            for i in range(self.N + 1):
                self.nodes[j*(self.N+1) + i, 0] = i/self.N
                self.nodes[j*(self.N+1) + i, 1] = j/self.N
        
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
        
        self.generate_mesh()
    
    
    def __repr__(self):
        return self.__class__.__name__ + '({!r})'.format(self.N)


    def __str__(self):
        string = repr(self) + '\n\nNodes:\n#\tx\ty\n'
        
        for j in range(self.N + 1):
            for i in range(self.N + 1):
                n = j*(self.N+1) + i
                string += (str(n) + '\t' +
                           "{:.5f}".format(self.nodes[n, 0]) + '\t' +
                           "{:.5f}".format(self.nodes[n, 1]) + '\n')
        
        string += '\nElements:\n#\tn1\tn2\tn3\n'
        
        for j in range(self.N):
            for i in range(self.N):
                n = 2*j*self.N + 2*i
                string += (str(n) + '\t' +
                           str(self.elems[n, 0]) + '\t' +
                           str(self.elems[n, 1]) + '\t' +
                           str(self.elems[n, 2]) + '\n')
                
                n = 2*j*self.N + 2*i+1
                string += (str(n) + '\t' +
                           str(self.elems[n, 0]) + '\t' +
                           str(self.elems[n, 1]) + '\t' +
                           str(self.elems[n, 2]) + '\n')
        
        return string
    
    
    def __len__(self):
        return len(self.elems)


    def __getitem__(self, position):
        return self.elems[position]

# End of Mesh class definition
        

##### Start of main program #####
        
mesh = Mesh(10)

