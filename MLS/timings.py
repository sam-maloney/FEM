#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Meshfree method simulation using moving least squares (MLS)

@author: Sam Maloney
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from MlsSim import MlsSim
from timeit import default_timer

# wavenumber for boundary function u(x,1) = g(x,y) = sinh(k*pi)
k = 1
def g(points):
    k = 1
    return np.sin(k*np.pi*points[:,0]) * np.sinh(k*np.pi*points[:,1])
            
# mls = MlsSim(10)
# mls.assembleStiffnessMatrix()

N=64
Nquad=1
form='quartic'
method='galerkin'
quadrature='uniform'
tolerance = 1e-10

# allocate arrays for convergence testing
start = 1.1
stop = 1.3
step = 0.1
nSamples = round((stop-start)/step) + 1
supports = np.linspace(start, stop, num=nSamples)
times = np.empty(nSamples, dtype='float64')
conds = np.empty(nSamples, dtype='float64')
fails = np.empty(nSamples, dtype='bool')
errors = np.empty(nSamples, dtype='float64')

# loop over N to test convergence where N is the number of
# grid cells along one dimension, each cell forms 2 triangles
# therefore number of nodes equals (N+1)*(N+1)
for iS, support in enumerate(supports):
    
    start_time = default_timer()
    
    # allocate arrays and compute boundary values
    mlsSim = MlsSim(N, g, Nquad, support, form, method, quadrature)
    
    # Assemble the stiffness matrix and solve for the approximate solution
    mlsSim.assembleStiffnessMatrix()
    mlsSim.solve(tol=tolerance, atol=tolerance)
    
    end_time = default_timer()
    
    # compute the analytic solution and error norms
    u_exact = g(mlsSim.nodes)
    # E_inf = np.linalg.norm(mlsSim.u - u_exact, np.inf)
    errors[iS] = np.linalg.norm(mlsSim.u - u_exact)/N
    
    times[iS] = end_time-start_time
    conds[iS] = mlsSim.cond()
    fails[iS] = (mlsSim.info != 0)
    
    print('support =', support)
    print('Condition Number =', conds[iS])
    print(f'Elapsed time = {end_time-start_time} s')
    
##### End of loop over supports #####

# supports_G = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3,
#        2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3. , 3.1, 3.2, 3.3, 3.4, 3.5])
# errors_G_c = np.array([7.73365617e-04, 7.30868645e-04, 6.56948039e-04, 5.75361761e-04,
#        5.18151051e-04, 5.07897017e-04, 4.29229371e-04, 3.07949936e-04,
#        2.55732775e-04, 3.35771771e-04, 5.42441322e-04, 8.46177611e-04,
#        1.24205809e-03, 1.70241828e-03, 2.20424607e-03, 2.77800109e-03,
#        3.52809077e-03, 4.58681837e-03, 6.15766257e-03, 8.57736705e-03,
#        1.26551488e-02, 1.99823519e-02, 3.35911277e-02, 6.39707033e-02,
#        1.45740477e-01])
# times_G_c = np.array([ 2.55538883,  2.53415708,  2.51122443,  2.59759859,  2.58615548,
#         2.65314941,  2.62434362,  2.61950575,  2.68502776,  2.69572755,
#         2.77162035,  2.98910332,  3.28325016,  3.95279311,  5.11618559,
#        12.1202988 , 15.14076699, 15.14265283, 15.12104725, 19.07310229,
#        19.56369004, 18.91503815, 18.91719603, 19.01263955, 18.85876551])
# conds_G_c = np.array([5.54880890e+04, 5.54751937e+04, 5.54538938e+04, 5.54325332e+04,
#        5.54208695e+04, 5.53702182e+04, 5.46122150e+04, 5.77386573e+04,
#        7.04570680e+04, 1.00626866e+05, 1.69140936e+05, 3.33915017e+05,
#        7.35718877e+05, 1.88800734e+06, 6.36342543e+06, 3.93743794e+07,
#        4.24331320e+08, 6.59020073e+08, 1.25224315e+08, 6.68840568e+07,
#        1.49154938e+08, 1.49912212e+09, 6.55408162e+08, 9.20755367e+07,
#        1.41141356e+08])
# fails_G_c = np.array([False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False,  True,  True,
#         True,  True,  True,  True,  True,  True,  True])
    
# errors_G_q = np.array([5.19086019e-04, 5.15959626e-04, 5.73608197e-04, 3.97704227e-04,
#         2.21269581e-04, 4.13270367e-04, 7.50718371e-04, 1.25765144e-03,
#         1.94265792e-03, 2.78503257e-03, 3.89516863e-03, 5.56634231e-03,
#         9.35243379e-03, 2.09297278e-02, 3.15948056e-02, 3.15427351e-02,
#         5.72437134e-02, 8.91007834e-02, 9.41282719e-02, 8.31888477e-02,
#         7.21600698e-02, 6.45993746e-02])
# times_G_q = np.array([ 2.60607194,  2.66251062,  2.74942154,  2.76613232,  2.76711235,
#         2.76992798,  2.86991612,  3.4698109 , 10.67953857,  7.06864877,
#         7.30408401,  9.43283859, 15.32809334, 15.32209693, 15.3201919 ,
#         15.23474473, 18.83929415, 19.00364715, 19.08652072, 19.3051175 ,
#         19.20002792, 19.48456151])
# conds_G_q = np.array([5.54209769e+04, 5.54319931e+04, 5.53839735e+04, 5.52489670e+04,
#         7.02311069e+04, 1.31415775e+05, 3.71569333e+05, 2.28296656e+06,
#         3.28932266e+08, 1.37637776e+07, 9.52833418e+06, 1.08989293e+07,
#         1.70198267e+07, 6.36593427e+07, 1.07127202e+08, 9.02837311e+06,
#         1.16860154e+07, 9.06547675e+06, 1.04893915e+07, 1.92904093e+07,
#         2.21314244e+07, 1.97079788e+07])
# fails_G_q = np.array([False, False, False, False, False, False, False, False,  True,
#         False, False, False,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True])

# supports_C = np.array([1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
#         2.8, 2.9, 3. ])
# errors_C_c = np.array([8.49958537E-01, 5.58828566E-01, 3.77868646E-01, 2.53769047E-01,
#         8.24589745E-02, 8.67944210E-02, 1.12225947E-01, 2.36910926E-01,
#         3.32407931E-01, 3.40898375E-01, 8.80199887E-02, 1.42015683E-01,
#         1.68331685E-01, 2.11418793E-01, 3.34909694E-01, 2.43119918E-01])
# times_C_c = np.array([7.28704831, 7.19890174, 7.80368475, 7.33962689, 3.19004136,
#         3.13762179, 3.70495732, 7.63920335, 8.82569128, 8.78560604,
#         3.87287002, 8.774823  , 8.76574467, 8.79760845, 9.44451105,
#         9.4731756 ])
# conds_C_c = np.array([2.33759289e+09, 1.68535670e+08, 1.16450373e+08, 1.72815424e+08,
#         2.44446204e+07, 2.21953273e+07, 2.01414617e+07, 2.33863166e+07,
#         3.07024466e+07, 2.51882168e+07, 2.51844050e+07, 1.12958576e+08,
#         4.05262184e+08, 2.18779789e+09, 6.42884362e+08, 1.02697984e+09])
# fails_C_c = np.array([ True,  True,  True,  True, False, False, False,  True,  True,
#         True, False,  True,  True,  True,  True,  True])

# errors_C_q = np.array([5.70576451E-01, 2.58221383E-01, 1.11047708E-01, 8.64858201E-02,
#         1.32614987E-01, 2.19299108E-01, 3.15817153E-01, 4.13913451E-01,
#         5.08345359E-01, 5.91870651E-01, 5.56528007E-01, 2.38666126E-01,
#         3.01908135E-01, 4.07124081E-01, 2.92438342E-01, 8.13446863E-01])
# times_C_q = np.array([7.2708704 , 7.24994387, 3.03880449, 3.0322526 , 3.0329999 ,
#         7.16458348, 7.79776608, 7.79655113, 8.9454295 , 9.07199307,
#         8.86134529, 3.78839149, 3.81958129, 8.87147831, 4.70290467,
#         9.57286041])
# conds_C_q = np.array([2.17538265E+09, 6.83135902E+07, 2.61663710E+07, 2.23589068E+07,
#         2.03911928E+07, 1.83524227E+08, 7.37481918E+07, 3.40220759E+08,
#         1.48351025E+09, 3.16348384E+07, 3.17774659E+07, 3.33008113E+07,
#         3.79620635E+07, 1.19597543E+08, 5.30241505E+07, 5.44961818E+07])
# fails_C_q = np.array([ True,  True, False, False, False,  True,  True,  True,  True,
#         True,  True, False, False,  True, False,  True])
    
    
# ##### Begin Plotting Routines #####

# # clear the current figure, if opened, and set parameters
# fig = plt.gcf()
# fig.clf()
# fig.set_size_inches(15,6)
# mpl.rc('axes', titlesize='xx-large', labelsize='x-large')
# mpl.rc('xtick', labelsize='large')
# mpl.rc('ytick', labelsize='large')
# plt.subplots_adjust(hspace = 0.3, wspace = 0.25)

# # plot the minimum E_2 error norms
# plt.subplot(131)
# plt.semilogy(supports_G, errors_G_c, '.-', label='Galerkin, cubic')
# plt.scatter(supports_G[fails_G_c], errors_G_c[fails_G_c], s=75, c='black', marker='x')
# plt.semilogy(supports_G, errors_G_q, '.-', label='Galerkin, quartic')
# plt.scatter(supports_G[fails_G_q], errors_G_q[fails_G_q], s=75, c='black', marker='x')
# plt.semilogy(supports_C, errors_C_c, '.-', label='Collocation, cubic')
# plt.scatter(supports_C[fails_C_c], errors_C_c[fails_C_c], s=75, c='black', marker='x')
# plt.semilogy(supports_C, errors_C_q, '.-', label='Collocation, quartic')
# plt.scatter(supports_C[fails_C_q], errors_C_q[fails_C_q], s=75, c='black', marker='x')
# plt.xticks([1.5, 2.0, 2.5, 3.0, 3.5])
# # plt.xlabel('support size [multiple of grid spacing]')
# plt.ylabel(r'minimum $E_2$ magnitude')
# plt.title('Error Magnitudes')
# # plt.legend(fontsize='x-large')

# # plot the times
# plt.subplot(132)
# plt.plot(supports_G, times_G_c, '.-', label='Galerkin, cubic')
# plt.scatter(supports_G[fails_G_c], times_G_c[fails_G_c], s=75, c='black', marker='x')
# plt.plot(supports_G, times_G_q, '.-', label='Galerkin, quartic')
# plt.scatter(supports_G[fails_G_q], times_G_q[fails_G_q], s=75, c='black', marker='x')
# plt.plot(supports_C, times_C_c, '.-', label='collocation, cubic')
# plt.scatter(supports_C[fails_C_c], times_C_c[fails_C_c], s=75, c='black', marker='x')
# plt.plot(supports_C, times_C_q, '.-', label='collocation, quartic')
# plt.scatter(supports_C[fails_C_q], times_C_q[fails_C_q], s=75, c='black', marker='x')
# plt.ylim([2.2, 27])
# plt.yticks(np.linspace(2.5, 25, 10))
# plt.xticks([1.5, 2.0, 2.5, 3.0, 3.5])
# plt.xlabel('support size [multiple of grid spacing]')
# plt.ylabel('elapsed time [s]')
# plt.title('Computation Times')
# plt.legend(fontsize='x-large', loc='upper left')

# # plot the condition number of the stiffness matrix
# plt.subplot(133)
# plt.semilogy(supports_G, conds_G_c, '.-', label='Galerkin, cubic')
# plt.scatter(supports_G[fails_G_c], conds_G_c[fails_G_c], s=75, c='black', marker='x')
# plt.semilogy(supports_G, conds_G_q, '.-', label='Galerkin, quartic')
# plt.scatter(supports_G[fails_G_q], conds_G_q[fails_G_q], s=75, c='black', marker='x')
# plt.semilogy(supports_C, conds_C_c, '.-', label='Collocation, cubic')
# plt.scatter(supports_C[fails_C_c], conds_C_c[fails_C_c], s=75, c='black', marker='x')
# plt.semilogy(supports_C, conds_C_q, '.-', label='Collocation, quartic')
# plt.scatter(supports_C[fails_C_q], conds_C_q[fails_C_q], s=75, c='black', marker='x')
# plt.xticks([1.5, 2.0, 2.5, 3.0, 3.5])
# # plt.xlabel('support size [multiple of grid spacing]')
# plt.ylabel('stiffness matrix condition number')
# plt.title('Condition Numbers')
# # plt.legend(fontsize='x-large')

# plt.savefig(f"MLS_{method}_{form}_{k}k_{Nquad}Q_{mlsSim.support*mlsSim.N}S.pdf",
#     bbox_inches = 'tight', pad_inches = 0)