
# Large-sCale Composite optimization
by Xiaocheng Tang [http://goo.gl/6QuMl]  

**lcc**, or Large-sCale Composite optimization, implements the algorithm **LHAC** -- <b> L</b>ow rank <b>H</b>essian <b>A</b>pproximation via <b>A</b>ctive-set <b>C</b>oordinate descent ([paper](http://goo.gl/ERZb3i))  -- for minimizing objective functions of the following form:

* `min f(x) + g(x)`  

where `f(x)` can be any _smooth_ function, i.e., _logistic loss_, _square loss_, etc., and `g(x)` is assumed to be _simple_, i.e., `l1-norm`, `l1/l2-norm`, etc.  In practice, the regularization functions `g(x)` are built into the software for users to choose from, and `f(x)` needs to be provided by users as function evaluation and gradient computation routines. 


## Installation
Clone this repo, but use the `GENERIC` branch
```bash
 git clone -b GENERIC https://github.com/LHAC/LHAC.git
 cd LHAC/src/
```
Modify lines at the beginning of the Makefile to indicate where your MATLAB is installed
```Makefile
## Contact your administrator if you do not know 
## where your matlab is installed.
MATLAB_ROOT = /Applications/MATLAB_R2013a.app
## on linux this should be $(MATLAB_ROOT)/bin/glnxa64 
MATLAB_LINK = $(MATLAB_ROOT)/bin/maci64
```
Run the provided Makefile from command line with optional arguments. For example,
* make mex file (.mexa64) on Linux and link to the MATLAB _BLAS_ library
```bash
make lhac.mexa64 
```
* make mex file (.mexmaci64) on Mac OSX and link to the _Accelerate BLAS_ library
```
make USE_ACCELERATE=true lhac.mexmaci64
```
* make command line tool and link to the _openBLAS_ library
```
make USE_OPENBLAS=true lhac.cmd
```
After the MEX-file is compiled successfully, start MATLAB in the same folder and run `demo_l1log.m` to verify the installation process. If successfully installed, `demo_l1log.m` will produce outputs on the screen and upon completion returns the classifier in the variable named `w`.
```
test 1 ---- reading data directly from file ---- 0%...
L1 - logistic
p = 123, N = 32561, nnz = 451592
test 1 ---- reading data directly from file ---- 100%!
test 2 ---- passing data from memory in MATLAB ---- 0%...
L1 - logistic
time = 2.6627e+00, iter = 431, optimal fval = 1.0559e+04
test 2 ---- passing data from memory in MATLAB ---- 100%!
```

## Extensions
**lcc** can be easily extended to handle functions other than `least square` or `logistic loss`.
As an example, **lcc** includes a class named `LogReg` that is derived from the base class `Objective` and implements the `logistic loss` function. The following lines of code demonstrate the use fo **lcc** for solving _sparse logistic regression_:
```c++
// instantiate a logistic loss object from the data file
LogReg* obj = new LogReg(param->fileName);

// create the algorithm object from the logistic function
LHAC<LogReg>* Alg = new LHAC<LogReg>(obj, param);

// solve the problem and return the solution
Solution* sols = Alg->solve();
```

In general, you can replace `LogReg.h` and `LogReg.cpp` with your own class files that implements the function `f(x)`. The only requirement is that **the class has to be derived from the base class `Objective` and implements its three member functions**:
```c++
template <typename Derived>
class Objective
{
public:
    // return the dimension of the problem
    inline unsigned long getDims() {
        return static_cast<Derived*>(this)->getDims();
    };
    
    // return function evaluation at the point wnew
    inline double computeObject(const double* wnew) {
        return static_cast<Derived*>(this)->computeObject(wnew);
    };
    
    // return gradient evaluation (in df) at point wnew
    inline void computeGradient(const double* wnew, double* df) {
        static_cast<Derived*>(this)->computeGradient(wnew, df);
    };
    
};
```


## Features
This package

* provides a fast and flexible framework for minimizing the sum of two functions
* provides an efficient implementation of sparse logistic regression and Lasso
* provides a fast _limited-memory BFGS_ library that can be used in general nonlinear optimizations 
* builds both a MATLAB interface and a standalone command line tool


## Citation
If you use LHAC in your research, please cite the following paper:

* Katya Scheinberg and Xiaocheng Tang, _Practical Inexact Proximal Quasi-Newton Method with Global Complexity Analysis_, submitted, 2014  ([BibTex](http://goo.gl/fVJgWN))


## Usage Guide

Refer to the file `demo_l1log.m`













