#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Meshfree method simulation

@author: Sam Maloney
"""

import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt

class MlsSim(object):
    """Node class for MLS method"""
    
    def __init__(self, N, Nquad, rho):
        self.N = N
        self.Nquad = Nquad
        self.rho = rho
        self.nodes = np.empty(((N+1)*(N+1), 2), dtype='float64')
        self.quads = np.empty((N*N*Nquad*Nquad, 2), dtype='float64')
    
    def __repr__(self):
        return self.__class__.__name__ + f"({self.N}, {self.rho})"

    def shape_function(self, pt, index, di, form):
        A = np.zeros((3,3), dtype='float64')
        dAdx = np.zeros((3,3), dtype='float64')
        dAdy = np.zeros((3,3), dtype='float64')
        
        