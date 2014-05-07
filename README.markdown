
# Lcc
by Xiaocheng Tang [http://goo.gl/6QuMl]  

**Lcc**, or LHAC for Composite minimization, implements the algorithm **LHAC** -- <b> L</b>ow rank <b>H</b>essian <b>A</b>pproximation in <b>A</b>ctive-set <b>C</b>oordinate descent ([paper](http://goo.gl/ERZb3i))  -- for minimizing composite functions, i.e.,  

* `min f(x) + g(x)`  

where `f(x)` can be any _smooth_ function, i.e., _logistic loss_, _square loss_, etc., and `g(x)` is assumed to be _simple_, i.e., `l1-norm`, `l1/l2-norm`, etc.  In practice, the regularization functions `g(x)` are built into the software for users to choose from, and `f(x)` needs to be provided by users as function evaluation and gradient computation routines. 

## Example
As an example, **Lcc** includes a class named `LogReg` that is derived from the base class `Objective` and implements the `logistic loss` function. The following lines of code demonstrate the use fo **Lcc** for solving _sparse logistic regression_:
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

* provides a fast and flexible framework for composite minimization
* provides, through the framework, an efficient implementation of sparse logistic regression
* provides a fast _limited-memory BFGS_ library that can be used in general nonlinear optimizations 
* builds both a MATLAB interface and a standalone command line tool


## Citation
If you use LHAC in your research, please cite the following paper:

* Katya Scheinberg and Xiaocheng Tang, _Practical Inexact Proximal Quasi-Newton Method with Global Complexity Analysis_, submitted, 2014  ([BibTex](http://goo.gl/fVJgWN))

## Build Guide
[Download](https://github.com/LHAC/LHAC/archive/GENERIC.zip) the package archive.

Extract the files:
```
tar xvf LHAC-GENERIC.zip
cd LHAC-GENERIC/src
```

Run the provided Makefile from command line:
```
make
make clean
```
and two files will be created in the current directory: the command line tool `LHACl1log` and the MATLAB MEX-file `LHACl1log.mexmaci64`.

> Note that the Makefile included is only intended for use on Mac OS X.  
> Note that the above procedures are tested successfully with Mac OS X 10.9.2 and MATLAB_R2013a


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













