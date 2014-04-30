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
#include <Accelerate/Accelerate.h>
#include "Lbfgs.h"
#include "Objective.h"


#define MAX_LENS 1024

#define __MATLAB_API__

enum { LIBSVM = 0, GENERAL };

enum { FULL, COL_VIEW, ROW_VIEW };

enum { LHAC_MSG_NO=0, LHAC_MSG_MIN, LHAC_MSG_NEWTON, LHAC_MSG_CD, LHAC_MSG_LINE, LHAC_MSG_MAX };

enum{ ALG_L1LOG = 1, ALG_SICS };

enum{  GREEDY= 1, STD };

typedef struct {
    double f;
    double g;
    double val; // f + g
    
    void add(double _f, double _g) {
        f = _f;
        g = _g;
        val = f + g;
    };
} Func;

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
    
    void releaseMe() {
        delete [] fval;
        delete [] normgs;
        delete [] t;
        delete [] niter;
        
        return;
    };
} solution;

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
    
} Parameter;



//solution* lhac(l1log* mdl);
solution* lhac(Objective* mdl, Parameter* param);



#endif /* defined(__LHAC_v1__lhac__) */
