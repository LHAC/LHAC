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
#include <vecLib/clapack.h>
#include <vecLib/cblas.h>
#include <math.h>

#define MAX_LENS 1024

#define __MATLAB_API__

enum { FULL, COL_VIEW, ROW_VIEW };

enum { LHAC_MSG_NO=0, LHAC_MSG_MIN, LHAC_MSG_NEWTON, LHAC_MSG_CD, LHAC_MSG_LINE, LHAC_MSG_MAX };

enum{ ALG_L1LOG = 1, ALG_SICS };

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
    double cdTime;
    double lsTime;
    double lbfgsTime1;
    double lbfgsTime2;
    unsigned long size; // max_newton_iter
} solution;

typedef struct {
    unsigned long i;
    unsigned long j;
    double vlt;
} ushort_pair_t;

typedef struct {
    ushort_pair_t* idxs; //p*(p+1)/2
    unsigned long* permut; //p
    unsigned long* idxs_vec_l; // vectorized lower
    unsigned long* idxs_vec_u; // vectorized upper
    unsigned long numActive;
    unsigned long _p_sics_;
} work_set_struct;

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
    
    /* line search */
    int sd_flag; // 1 = sufficient decrease; 0 = line search
    
    /* gama in lbfgs */
    double shrink = 1; // gama = gama/shrink
} command_line_param;





#endif /* defined(__LHAC_v1__lhac__) */
