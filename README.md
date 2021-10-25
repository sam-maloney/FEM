# Samuel Maloney - University of Warwick

## The Problem

My program solves a simple 2D Poisson problem, specifically the homogeneous equation,

> <a href="https://www.codecogs.com/eqnedit.php?latex=-u_{xx}(x,y)-u_{yy}(x,y)=0,\qquad(x,y)\in\Omega=(0,1)\times(0,1)," target="_blank"><img src="https://latex.codecogs.com/svg.latex?-u_{xx}(x,y)-u_{yy}(x,y)=0,\qquad(x,y)\in\Omega=(0,1)\times(0,1)," title="-u_{xx}(x,y)-u_{yy}(x,y)=0,\qquad(x,y)\in\Omega=(0,1)\times(0,1)," /></a>

with Dirichlet boundary conditions given as,

> <a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}u(0,y)=u(1,y)=0,&\qquad0\leq&space;y\leq1,\\u(x,0)=0,&\qquad0\leq&space;x\leq1,\\u(x,1)=\sin(k\pi&space;x),&\qquad0<x<1,\end{align*}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\begin{align*}u(0,y)=u(1,y)=0,&\qquad0\leq&space;y\leq1,\\u(x,0)=0,&\qquad0\leq&space;x\leq1,\\u(x,1)=\sin(k\pi&space;x),&\qquad0<x<1,\end{align*}" title="\begin{align*}u(0,y)=u(1,y)=0,&\qquad0\leq y\leq1,\\u(x,0)=0,&\qquad0\leq x\leq1,\\u(x,1)=\sin(k\pi x),&\qquad0<x<1,\end{align*}" /></a>

which has analytic solutions given by,

> <a href="https://www.codecogs.com/eqnedit.php?latex=u_k(x,y)=\sin(k\pi&space;x)\sinh(k\pi&space;y)." target="_blank"><img src="https://latex.codecogs.com/svg.latex?u_k(x,y)=\sin(k\pi&space;x)\sinh(k\pi&space;y)." title="u_k(x,y)=\sin(k\pi x)\sinh(k\pi y)." /></a>

## The Program

The numerical solution is computed using linear finite elements (FEM) on a regular triangular mesh. The script defines a simple mesh class which is used to generate and store such a mesh for a given number of grid divisions. The program then loops over the elements to compute and assemble the FEM stiffness matrix and uses some sparse linear algebra routines from scipy to compute the numerical solution for the given mesh.

A loop over the number of grid divisions is performed, doubling each time, such that the convergence of the numerical solution towards the expected analytic result can be observed. A figure is produced showing the final numeric solution side by side with the analytic solution for visual comparison, as well as plots of the error norms and observed order of convergence to validate the expected 2nd order spatial accuracy of the method.
