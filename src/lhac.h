//
//  lhac.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 1/31/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#ifndef __LHAC_v1__lhac__
#define __LHAC_v1__lhac__


#define MAX_LENS 1024

#if defined(LANG_M) || defined(MATLAB_MEX_FILE)
#include <mex.h>
#define MSG mexPrintf
#endif

#ifndef MSG
#define MSG printf
#endif

//#define __MATLAB_API__



enum { LHAC_MSG_NO=0, LHAC_MSG_MIN, LHAC_MSG_NEWTON, LHAC_MSG_CD, LHAC_MSG_LINE, LHAC_MSG_MAX };


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
    unsigned long record1; // count #factorizations
    
    unsigned long ngval;
    unsigned long nfval;
    unsigned long nls; // # of line searches
    double gvalTime;
    double fvalTime;
    
    /* result */
    double* w;
    int p_sics; //dimension of w
} solution;

inline void releaseSolution(solution* sols)
{
    delete [] sols->fval;
    delete [] sols->normgs;
    delete [] sols->t;
    delete [] sols->niter;
    delete [] sols->w;
    delete sols;
    
    return;
}


typedef struct {
    char* fileName;
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
    double shrink; // gama = gama/shrink
    
    /* rho in sd condition */
    double rho;
    
    /* max_cdpass = 1 + iter / cdrate */
    /* best empirical value = 15  */
    unsigned long cd_rate;
} param;

typedef struct {
    char* fileName;
    unsigned short max_iter;
    double lmd;
    double opt_outer_tol;
    int dense; // default no (0), using libsvm format
    int randomData; // default no, reading data from fileNmae
    unsigned long random_p; // random data features
    unsigned long random_N; // random data numbers
    double nnz_perc;
    int verbose; // LHAC_MSG_NO, etc.
    int alg; // ALG_L1LOG = 1, ALG_SICS
    
    /* line search */
    int sd_flag; // 1 = sufficient decrease; 0 = line search
    
    /* gama in lbfgs */
    double shrink; // gama = gama/shrink
    
    /* rho in sd condition */
    double rho;

} command_line_param;





#endif /* defined(__LHAC_v1__lhac__) */
