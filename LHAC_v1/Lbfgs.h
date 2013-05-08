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

#define MAX_SY_PAIRS 100


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
    double* buff; // p
    
    // for test
    double tQ = 0;
    double tR = 0;
    double tQ_bar = 0;
    
    LBFGS(unsigned long _p, unsigned short _l, double _s);
    
    ~LBFGS();
    
    void initData(double* w, double* w_prev, double* L_grad, double* L_grad_prev);
    
    void computeLowRankApprox();
    
    void computeLowRankApprox(work_set_struct* work_set);
    
    void computeLowRankApprox_v2(work_set_struct* work_set);
    
    void updateLBFGS(double* w, double* w_prev, double* L_grad, double* L_grad_prev);
    
    void computeHDiag(double* H_diag);
    
    double computeHdj(double Di, double* d_bar, unsigned long idx);
    
    void updateDbar(double* d_bar, unsigned long idx, double z);
    
private:
    LMatrix* Sm;
    LMatrix* Tm;
    LMatrix* Lm;
    LMatrix* STS;
    double* Dm;
    double* R;
    unsigned long p; // no. of rows in Q
    unsigned short l; // lbfgs param
    
    double shrink;
    
    /* for lapack */
    int ipiv[MAX_SY_PAIRS+1];
    int lwork = MAX_SY_PAIRS*MAX_SY_PAIRS;
    double work[MAX_SY_PAIRS*MAX_SY_PAIRS];
    
    void computeQR();
    void computeQR_v2(work_set_struct* work_set);
//    unsigned long* Q_bar_idxs; // cols of Q_bar to update
    
};
    

#endif /* defined(__LHAC_v1__Lbfgs__) */
