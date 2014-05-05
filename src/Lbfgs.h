//
//  Lbfgs.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 1/30/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#ifndef __LHAC_v1__Lbfgs__
#define __LHAC_v1__Lbfgs__

typedef struct {
    unsigned long i;
    unsigned long j;
    double vlt;
} ushort_pair_t;

struct work_set_struct {
    ushort_pair_t* idxs; //p*(p+1)/2
    unsigned long* permut; //p
    unsigned long* idxs_vec_l; // vectorized lower
    unsigned long* idxs_vec_u; // vectorized upper
    unsigned long numActive;
    unsigned long _p_sics_;
    
    work_set_struct(unsigned long p) {
        idxs = new ushort_pair_t[p];
        permut = new unsigned long[p];
    }
    
    ~work_set_struct() {
        delete [] idxs;
        delete [] permut;
    }
};

class LMatrix {
public:
    double** data;
    double* data_space;
    unsigned long rows;
    unsigned short cols;
    unsigned long maxrows;
    unsigned short maxcols;
    
    LMatrix(const unsigned long s1, const unsigned long s2); // initiate to be a matrix s1 X s2
    
    ~LMatrix();
    
    void init(const double* const x, const unsigned long n1, const unsigned short n2);
    // initialized to be the matrix n1 X n2
    
    void print();
    
    inline void insertRow(const double* const x); // to the bottom
    
    inline void insertCol(const double* const x); // to the rightmost
    
    inline void deleteRow(); // first
    
    inline void deleteCol(); // leftmost
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
    
    LBFGS(const unsigned long _p, const unsigned short _l, const double _s);
    
    ~LBFGS();
    
    void initData(const double* const w, const double* const w_prev,
                  const double* const L_grad, const double* const L_grad_prev);
    
    void computeLowRankApprox_v2(work_set_struct* work_set);
    
    void updateLBFGS(const double* const w, const double* const w_prev,
                     const double* const L_grad, const double* const L_grad_prev);
    
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
