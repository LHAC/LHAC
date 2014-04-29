
# LHAC
by Xiaocheng Tang [http://goo.gl/6QuMl]  

**LHAC** implements the algorithm -- <b> L</b>ow rank <b>H</b>essian <b>A</b>pproximation in <b>A</b>ctive-set <b>C</b>oordinate descent ([paper](http://goo.gl/ERZb3i))  -- for minimizing composite functions, i.e.,  

* `min f(x) + g(x)`  

where `f(x)` can be any _smooth_ function, i.e., _logistic loss_, _square loss_, etc., and `g(x)` is assumed to be _simple_, i.e., `l1-norm`, `l1/l2-norm`, etc.  In practice, the regularization functions will be built into the software for users to choose from, and users will need to provide the function evaluation and gradient computation routines for `f(x)`.   There are, for now, two varieties of LHAC that handle

* _sparse inverse covariance selections_ problems 
* _sparse logistic regression_ problems

This package, in particular, implements LHAC for solving _sparse inverse covariance selection_ problems, and recovers from a low-rank sample covariance matrix the inverse covariance matrix that is expected to have a sparse structure.

On use of LHAC for general composite minimization, please contact me [here](http://goo.gl/6QuMl) for more details.

## Features
This package

* handles _sparse inverse covariance selections_ problems
* supports various platforms, i.e., Mac OS X and Linux
* supports both BLAS and CBLAS interfaces
* includes a fast _limited-memory BFGS_ library that can be used in general nonlinear optimizations 


## Citation
If you use LHAC in your research, please cite the following paper:

* Katya Scheinberg and Xiaocheng Tang, _Practical Inexact Proximal Quasi-Newton Method with Global Complexity Analysis_, submitted, 2014  ([BibTex](http://goo.gl/fVJgWN))

## Build Guide
[Download](http://goo.gl/wuFEJ4) the package archive.


Extract the files:
```
tar xvf LEHIGH-universal.zip
cd LEHIGH-universal/src
```

LHAC comes with a MATLAB interface through MEX-files. To build the MEX-file on Linux, just run
```
mex -largeArrayDims sics_lhac.cpp sics_lhac-mex.cpp Lbfgs.cpp  -lmwblas -lmwlapack -lrt -output LHAC
```
Or if you are running Mac OS, you may compile the program using the provided **Makefile** (need to modify the first line to reflect where MATLAB is installed). Note that LHAC uses BLAS and LAPACK. The above command links to the BLAS and LAPACK library come with MATLAB, and the Makefile links to Apple's Accelerate framework that contains a version of BLAS and LAPACK optimized for Mac OS. 

You will probably also need to modify the `mexopts.sh` in `~/.matlab` before you run `mex` so that the compiler uses c++11 standard. To do that, simply replace the flag `-ansi` in CXXFLAG with `-std=c++0x`.

## Usage Guide

After the MEX-file is compiled successfully, start MATLAB in the same folder and run `LHAC_demo.m` to verify the installation process. If successfully installed, `LHAC_demo.m` will produce outputs on the screen and upon completion returns the inverse covariance matrix in the variable named `W`.

Typical usage of LHAC is:
```
W = LHAC(S, lambda, Param);
```
where the solution `W` is the inverse covariance matrix recovered from the input `S` the sample covariance matrix, `lambda` is a positive scalar known as the regularization parameter and `Param` is a MATLAB `struct` that contains the algorithm parameters for LHAC. Some commonly-used parameters are listed below:

* `v`: verbosity level 0-3 (default 2)
* `e`: optimality tolerance (default 1e-6)
* `i`: maximum number of iterations allowed (default 500)

Optionally, records of the optimization process, i.e., objective values, iteration counter, norm of the subgradient, etc., can be passed out to the output variable list besides the optimal solution:
```
[W, iter, fval, t, normgs, numActive] = LHAC(S, lambda, Param);
```













