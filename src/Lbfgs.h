//
//  Lbfgs.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 1/30/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#ifndef __LHAC_v1__Lbfgs__
#define __LHAC_v1__Lbfgs__

#include "lhac.h"


class LMatrix {
public:
    double** data;
    double* data_space;
    unsigned long rows;
    unsigned short cols;
    unsigned long maxrows;
    unsigned short maxcols;
    
    LMatrix(unsigned long s1, unsigned long s2); // initiate to be a matrix s1 X s2
    
    ~LMatrix();
    
    void init(double* x, unsigned long n1, unsigned short n2);
    // initialized to be the matrix n1 X n2
    
    void print();
    
    void insertRow(double* x); // to the bottom
    
    void insertCol(double* x); // to the rightmost
    
    void deleteRow(); // first
    
    void deleteCol(); // leftmost
};

class LBFGS {
public:
    double* Q;
    double* Q_bar;
    unsigned short m; // no. of cols in Q
    double gama;
    
    // for test
    double tQ;
    double tR;
    double tQ_bar;
    
    double* buff;
    
    double shrink;
    
    LBFGS(unsigned long _p, unsigned short _l, double _s);
    
    ~LBFGS();
    
    void initData(double* w, double* w_prev, double* L_grad, double* L_grad_prev);
    
    void computeLowRankApprox_v2(work_set_struct* work_set);
    
    void updateLBFGS(double* w, double* w_prev, double* L_grad, double* L_grad_prev);
    
private:
    LMatrix* Sm;
    LMatrix* Tm;
    LMatrix* Lm;
    LMatrix* STS;
    unsigned long* permut; // for updating lbfgs, length of l
    double* permut_mx; // for updating lbfgs, l*l
    double* buff2; // length of l
    
    double* Dm;
    double* R;
    unsigned long p; // no. of rows in Q
    unsigned short l; // lbfgs param
    
    
    void computeQR_v2(work_set_struct* work_set);
    
};
    

#endif /* defined(__LHAC_v1__Lbfgs__) */
