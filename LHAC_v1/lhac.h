//
//  lhac.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 1/31/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#ifndef __LHAC_v1__lhac__
#define __LHAC_v1__lhac__

#include <time.h>
//#include <vecLib/clapack.h>
#include <Accelerate/Accelerate.h>
//#include <vecLib/cblas.h>
#include <math.h>
#include "Lbfgs.h"


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#define MAX_LENS 1024

#define __MATLAB_API__

enum { LIBSVM = 0, GENERAL };

enum { FULL, COL_VIEW, ROW_VIEW };

enum { LHAC_MSG_NO=0, LHAC_MSG_MIN, LHAC_MSG_NEWTON, LHAC_MSG_CD, LHAC_MSG_LINE, LHAC_MSG_MAX };

enum{ ALG_L1LOG = 1, ALG_SICS };

enum{  GREEDY= 1, STD };

typedef struct {
    int index; // starting from 1 (not 0), ending with -1
    double value;
} feature_node;

typedef struct {
    feature_node** X;
    double* y;
    unsigned long p;
    unsigned long N;
    unsigned long nnz; // number of nonzeros
    feature_node* x_space;
} training_set_sp;

typedef struct {
    double* X;
    double* y;
    unsigned long p;
    unsigned long N;
} training_set;

typedef struct {
    double* t;
    double* fval;
    double* normgs;
    int* niter;
    unsigned long* numActive;
    double cdTime;
    double lsTime;
    double lbfgsTime1;
    double lbfgsTime2;
    unsigned long size; // max_newton_iter
    
    unsigned long ngval;
    unsigned long nfval;
    unsigned long nls; // # of line searches
    double gvalTime;
    double fvalTime;
    
    /* result */
    int p_sics; //dimension of w
} solution;


typedef struct {
    unsigned long work_size;
    unsigned short max_iter;
    unsigned long max_inner_iter;
    double* lmd;
    double opt_inner_tol;
    double opt_outer_tol;
    /**** line search ****/
    double bbeta;
    double ssigma;
    unsigned long max_linesearch_iter;
    unsigned long l; // lbfgs sy pair number <= MAX_SY_PAIRS
    int verbose; //
    
    /* line search */
    int sd_flag; // 1 = sufficient decrease; 0 = line search
    
    /* gama in lbfgs */
    double shrink = 1; // gama = gama/shrink
} param;

typedef struct {
    char* fileName;
    unsigned short max_iter;
    double lmd;
    int dense; // default no (0), using libsvm format
    int randomData; // default no, reading data from fileNmae
    unsigned long random_p; // random data features
    unsigned long random_N; // random data numbers
    double nnz_perc;
    int verbose; // LHAC_MSG_NO, etc.
    int alg = ALG_L1LOG; // ALG_L1LOG = 1, ALG_SICS
    double opt_outer_tol;
    
    /* line search */
    int sd_flag; // 1 = sufficient decrease; 0 = line search
    
    /* gama in lbfgs */
    double shrink = 1; // gama = gama/shrink
    
    double rho;
} command_line_param;



typedef struct {
    char* fileName;
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
    
    /* line search */
    int sd_flag; // 1 = sufficient decrease; 0 = line search
    
    /* gama in lbfgs */
    double shrink = 1; // gama = gama/shrink
    
    double rho;
    
    unsigned long cd_rate;
    
    // active set stragety
    unsigned long active_set;
} l1log_param;


class l1log
{
public:
    void coordinateDsecent(LBFGS* lR, work_set_struct* work_set);
    
    double suffcientDecrease(LBFGS* lR, work_set_struct* work_set, double mu, solution* sols);
    
    double computeModelValue(LBFGS* lR, work_set_struct* work_set, double step_size);
    
    double computeSubgradient();
    
    double computeObject(double* wnew); // cache e_ywx;
    
    double computeObject(double* wnew, double _a); // update e_ywx using a and Xd
    double computeObject(); // w = w + D;
    
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
    
    double* D;
    double normsg0;
    double f_current;
    double* w_prev;
    double* w;
    double* L_grad_prev;
    double* L_grad;
    
    double dQ;
    
    unsigned long iter;
    
    int MSG = LHAC_MSG_MAX;
    
    double timeBegin;
    
    unsigned long p;
    unsigned long N;
    
private:
    double* H_diag; // p
    double* d_bar; // 2*l
    double* e_ywx; // N
    double* B; // N
    double* Xd; // N
    
    double* L_prob; // probability of choosing ith coordinate; p
    
    
    /**** data general format ****/
    double* X;
    /**** data libsvm format ****/
    feature_node** X_libsvm;
    feature_node* x_space;
    
    double* y;
    
    int mode; // data format { LIBSVM, GENERAL }
    
    void greedySelector( work_set_struct* &work_set );
    
    void stdSelector( work_set_struct* &work_set );
    
    unsigned long randomCoordinateSelector(unsigned long range);
    
    void initData(training_set* Dset);
    
    void initData(training_set_sp* Dset);
    
    void initVars();
    
    void init(training_set* Dset, l1log_param* _param);
    
    void init(training_set_sp* Dset, l1log_param* _param);
};


solution* lhac(l1log* mdl);



#endif /* defined(__LHAC_v1__lhac__) */
