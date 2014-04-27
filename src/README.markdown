
# LHAC
by Xiaocheng Tang [http://goo.gl/6QuMl]  

**LHAC** implements the regularized empirical risk minimization algorithm -- <b> L</b>ow rank <b>H</b>essian <b>A</b>pproximation in <b>A</b>ctive-set <b>C</b>oordinate descent ([paper](http://goo.gl/ERZb3i))  --  and is optimized specifically for solving _sparse inverse covariance selection_ problems which recover from a low-rank sample covariance matrix the inverse covariance matrix that is expected to have a sparse structure.

## Features
LHAC

* can handle _sparse inverse covariance selections_ problems
* can handle _sparse logistic regression_ problems (please [contact me](mailto:xiaocheng.t@gmail.com) for more details)
* supports various platforms, i.e., Mac OS X and Linux
* supports both BLAS and CBLAS interfaces

The extension of LHAC to other regularizations other than l1 is possible. In particular, the one that handles **group sparsity** is forthcoming.


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