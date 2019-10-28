# -*- coding: utf-8 -*-

import numpy as np

class Mesh:
    def __init__(self, N):
        elems = np.empty((3,2*N*N))
        nodes = np.empty((2,(N+1)*(N+1)))
