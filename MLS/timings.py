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
Nquad=3
form='cubic'
method='galerkin'
quadrature='uniform'
tolerance = 1e-10

# allocate arrays for convergence testing
start = 1.1
stop = 2.5
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
    
    print('support =', support)
    
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
    
    print('Condition Number =', conds[iS])
    print(f'Elapsed time = {end_time-start_time} s')
    
##### End of loop over supports #####

supports = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3,
       2.4, 2.5])

errors_1 = np.array([7.73365617e-04, 7.30868645e-04, 6.56948039e-04, 5.75361762e-04,
       5.18151051e-04, 5.07897035e-04, 4.29229525e-04, 3.07949590e-04,
       2.55732775e-04, 3.35771771e-04, 5.42441322e-04, 8.46177611e-04,
       1.24205809e-03, 1.70241828e-03, 2.20424607e-03])
times_1 = np.array([2.53707446, 2.50077454, 2.51370958, 2.5209281 , 2.53166216,
       2.65447504, 2.66248749, 2.67650774, 2.72066498, 2.71211688,
       2.76807121, 3.02141176, 3.28819224, 3.81241878, 4.95883371])
conds_1 = np.array([  55488.08902459,   55475.19370667,   55453.8938353 ,
         55432.53315276,   55420.86946244,   55370.21815542,
         54612.21496248,   57738.65726404,   70457.06800962,
        100626.86623669,  169140.93625616,  333915.01695902,
        735718.87668048, 1888007.34305478, 6363425.43388948])
fails_1 = np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False])
    
errors_2_u = np.array([1.78440271e-04, 3.65701974e-04, 4.16549243e-04, 3.46276384e-04,
       2.33185216e-04, 1.49597711e-04, 1.28613977e-04, 9.78422277e-05,
       7.00566281e-05, 7.17964967e-05, 9.07608764e-05, 1.09280144e-04,
       1.86901846e-04, 3.39001301e-04, 5.05791171e-04])
times_2_u = np.array([ 6.87369151,  6.77246112,  6.8713262 ,  6.88661014,  6.85343989,
        6.8220318 ,  6.86871876,  6.99170431,  6.98953464,  7.07910309,
        7.16537224,  7.23208659,  7.51826483,  8.23897757, 10.13909895])
conds_2_u = np.array([  69425.11227786,   48817.9918618 ,   45392.52679079,
         43953.62368207,   38681.78457307,   31264.84498601,
         25151.42808312,   21149.66806804,   18700.12575489,
         19002.45947185,   26932.35898247,   59005.34541492,
        182783.18509681,  798248.69009849, 7109691.81890652])
fails_2_u = np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False])

errors_2_g = np.array([3.99459067e-04, 6.25934254e-05, 2.23982374e-04, 2.22185636e-04,
       1.16454191e-04, 1.40102550e-04, 1.65271249e-04, 1.24586276e-04,
       5.73547449e-05, 5.47317206e-05, 1.13568852e-04, 1.39874637e-04,
       1.21827167e-04, 7.24282193e-05, 7.34745357e-05])
times_2_g = np.array([ 6.90978239,  6.80307931,  6.8658608 ,  6.88721512,  6.86744588,
        6.84065929,  6.85878397,  6.93353788,  7.08221337,  7.1858533 ,
        7.199476  ,  7.34561773,  7.57223439,  8.28927702, 10.40217116])
conds_2_g = np.array([   78107.85786083,    62722.65214896,    52992.20112116,
          46795.99764083,    37830.39372121,    29160.4403467 ,
          23211.89013014,    19577.90521941,    17626.89628457,
          18233.1638543 ,    25392.94488348,    54121.90734509,
         175821.57737143,   910026.12721674, 12401951.48844856])
fails_2_g = np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False])

errors_3_u = np.array([1.33283951e-05, 2.25719527e-04, 2.98745827e-04, 2.59172860e-04,
       1.78514333e-04, 1.35068157e-04, 1.03381350e-04, 7.92804780e-05,
       6.30635174e-05, 4.99108537e-05, 5.43138555e-05, 6.62812765e-05,
       1.12154045e-04, 1.88862214e-04, 2.43569614e-04])
times_3_u = np.array([13.82271837, 13.83271186, 13.86750441, 13.93780931, 14.01114955,
       14.09844047, 14.2109136 , 14.43052099, 14.4651814 , 14.59945066,
       14.68159995, 14.98187826, 15.25491772, 16.14956953, 17.13984868])
conds_3_u = np.array([  62288.51573444,   53660.45076088,   50706.36323264,
         44270.02165549,   36596.59073818,   30070.87434458,
         24821.93929637,   20905.66085204,   18680.05965649,
         19490.71976328,   28224.29510861,   59963.84818014,
        173098.03378692,  697656.96800833, 5079397.6100785 ])
fails_3_u = np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False])
    
    
##### Begin Plotting Routines #####

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(15,6)
mpl.rc('axes', titlesize='xx-large', labelsize='x-large')
mpl.rc('xtick', labelsize='large')
mpl.rc('ytick', labelsize='large')
plt.subplots_adjust(hspace = 0.3, wspace = 0.25)

# plot the minimum E_2 error norms
plt.subplot(131)
plt.semilogy(supports, errors_1, '.-', label='1, uniform')
plt.scatter(supports[fails_1], errors_1[fails_1], s=75, c='black', marker='x')
plt.semilogy(supports, errors_2_u, '.-', label=r'$2\times2$, uniform')
plt.scatter(supports[fails_2_u], errors_2_u[fails_2_u], s=75, c='black', marker='x')
plt.semilogy(supports, errors_2_g, '.-', label=r'$2\times2$, Gauss-Legendre')
plt.scatter(supports[fails_2_g], errors_2_g[fails_2_g], s=75, c='black', marker='x')
plt.semilogy(supports, errors_3_u, '.-', label=r'$3\times3$, uniform')
plt.scatter(supports[fails_3_u], errors_3_u[fails_3_u], s=75, c='black', marker='x')
plt.xticks([1.0, 1.5, 2.0, 2.5])
# plt.xlabel('support size [multiple of grid spacing]')
plt.ylabel(r'$E_2$ magnitude')
plt.title('Error Magnitudes')
# plt.legend(fontsize='x-large')

# plot the times
plt.subplot(132)
plt.plot(supports, times_1, '.-', label='1, uniform')
plt.scatter(supports[fails_1], times_1[fails_1], s=75, c='black', marker='x')
plt.plot(supports, times_2_u, '.-', label=r'$2\times2$, uniform')
plt.scatter(supports[fails_2_u], times_2_u[fails_2_u], s=75, c='black', marker='x')
plt.plot(supports, times_2_g, '.-', label=r'$2\times2$, Gauss-Legendre')
plt.scatter(supports[fails_2_g], times_2_g[fails_2_g], s=75, c='black', marker='x')
plt.plot(supports, times_3_u, '.-', label=r'$3\times3$, uniform')
plt.scatter(supports[fails_3_u], times_3_u[fails_3_u], s=75, c='black', marker='x')
# plt.ylim([2.2, 24])
plt.yticks(np.linspace(2.5, 17.5, 7))
plt.xticks([1.0, 1.5, 2.0, 2.5])
plt.xlabel('support size [multiple of grid spacing]')
plt.ylabel('elapsed time [s]')
plt.title('Computation Times')
# plt.legend(fontsize='x-large', loc='upper left')

# plot the condition number of the stiffness matrix
plt.subplot(133)
plt.semilogy(supports, conds_1, '.-', label='1, uniform')
plt.scatter(supports[fails_1], conds_1[fails_1], s=75, c='black', marker='x')
plt.semilogy(supports, conds_2_u, '.-', label=r'$2\times2$, uniform')
plt.scatter(supports[fails_2_u], conds_2_u[fails_2_u], s=75, c='black', marker='x')
plt.semilogy(supports, conds_2_g, '.-', label=r'$2\times2$, Gauss-Legendre')
plt.scatter(supports[fails_2_g], conds_2_g[fails_2_g], s=75, c='black', marker='x')
plt.semilogy(supports, conds_3_u, '.-', label=r'$3\times3$, uniform')
plt.scatter(supports[fails_3_u], conds_3_u[fails_3_u], s=75, c='black', marker='x')
plt.xticks([1.0, 1.5, 2.0, 2.5])
# plt.xlabel('support size [multiple of grid spacing]')
plt.ylabel('stiffness matrix condition number')
plt.title('Condition Numbers')
plt.legend(fontsize='x-large', loc='upper left')

# plt.savefig(f"MLS_timings_quadrature_1k.pdf",
#     bbox_inches = 'tight', pad_inches = 0)