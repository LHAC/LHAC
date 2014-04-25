
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
mex -largeArrayDims sics_lhac.cpp sics_lhac-mex.cpp Lbfgs.cpp  -lmwblas -lmwlapack -lrt
```

You will probably also need to modify the `mexopts.sh` in `~/.matlab` before you run `mex` so that gcc uses c++11 standard. To do that, simply replace the flag `-ansi` in CXXFLAG with `-std=c++0x`.

After you have the MEX-file. Simply start matlab in the same folder and run demo1.m. You should be also to see the outputs immediately and have a structure returned named algs.


