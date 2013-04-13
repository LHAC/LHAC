//
//  l1log.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 1/27/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#ifndef __LHAC_v1__l1log__
#define __LHAC_v1__l1log__

#include <iostream>
#include "lhac.h"
#include "Lbfgs.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

enum { LIBSVM = 0, GENERAL };

typedef struct {
    unsigned long work_size;
    unsigned short max_iter;
    unsigned long max_inner_iter;
    double lmd;
    double opt_inner_tol;
    double opt_outer_tol;
    /**** line search ****/
    double bbeta;
    double ssigma;
    unsigned long max_linesearch_iter;
    unsigned long l; // lbfgs sy pair number <= MAX_SY_PAIRS
    int verbose;
} l1log_param;


class l1log
{
public:
    // H_diag, d_bar and D be allocated and initialized before function call
    void coordinateDescent(double* Q_bar, double* Q,
                           double gama, work_set_struct* work_set,
                           unsigned short m);
    void coordinateDsecent(LBFGS* lR, work_set_struct* work_set);
    
    double computeSubgradient();
    
    double computeObject(double* wnew); // cache e_ywx;
    
    double computeObject(double* wnew, double _a); // update e_ywx using a and Xd
    
    void lineSearch();
    
    void computeGradient();
    
    void computeWorkSet( work_set_struct* &work_set );  
    
    l1log(training_set* Dset);
    
    l1log(training_set* Dset, l1log_param* _param);
    
    l1log(training_set_sp* Dset, l1log_param* _param);
    
    l1log(training_set_sp* Dset);
    
    ~l1log();
    
    /* Parameters */
    l1log_param* param;
    
    double normsg0;
    double f_current;
    double* w_prev;
    double* w;
    double* L_grad_prev;
    double* L_grad;
    
    unsigned long iter;
    
    int MSG = LHAC_MSG_MAX;
    
    double timeBegin;
    
    unsigned long p;
    unsigned long N;
    
private:
    double* D;
    double* H_diag; // p
    double* d_bar; // 2*l
    double* e_ywx; // N
    double* B; // N
    double* Xd; // N
    
    
    /**** data general format ****/
    double* X;
    /**** data libsvm format ****/
    feature_node** X_libsvm;
    feature_node* x_space; 
    
    double* y;
    
    int mode; // data format { LIBSVM, GENERAL }
    
    void initData(training_set* Dset);
    
    void initData(training_set_sp* Dset);
    
    void initVars();
    
    void init(training_set* Dset, l1log_param* _param);
    
    void init(training_set_sp* Dset, l1log_param* _param);
};

#endif /* defined(__LHAC_v1__l1log__) */
