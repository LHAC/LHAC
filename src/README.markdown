
# LHAC
by Xiaocheng Tang [http://goo.gl/6QuMl]  

**LHAC** implements the regularized empirical risk minimization algorithm -- <b> L</b>ow rank <b>H</b>essian <b>A</b>pproximation in <b>A</b>ctive-set <b>C</b>oordinate descent ([paper](http://goo.gl/ERZb3i))  --  and is optimized specifically for solving _sparse inverse covariance selection_ problems.


## Citation
If you use LHAC in your research, please cite the following paper:

* "Practical Inexact Proximal Quasi-Newton Method with Global Complexity Analysis", Katya Scheinberg, Xiaocheng Tang ([BibTex]())

## Build Guide
[Download](http://goo.gl/wuFEJ4) the package archive.


Extract the files:
```
tar xvf LEHIGH-universal.zip
cd LEHIGH-universal/src
```

LHAC comes with a MATLAB interface through MEX-files. To build the MEX-file, just run
```
mex -largeArrayDims sics_lhac.cpp sics_lhac-mex.cpp Lbfgs.cpp  -lmwblas -lmwlapack -lrt -output LHAC
```
Note that LHAC uses BLAS and LAPACK. The above command links to the BLAS and LAPACK library come with MATLAB. You can also link to one tuned for your platform, i.e., use `-framework Accelerate` on Mac OS to link to Accelerate. In that case, you might also want to take a look at the header file `liblapack.h` provided in the LHAC package.

You will probably also need to modify the `mexopts.sh` in `~/.matlab` before you run `mex` so that gcc uses c++11 standard. To do that, simply replace the flag `-ansi` in CXXFLAG with `-std=c++0x`.

## Usage Guide

After you have the MEX-file. Simply start MATLAB in the same folder and run `LHAC_demo.m`. You should be able to see the outputs immediately and have a structure returned named `algs`.


