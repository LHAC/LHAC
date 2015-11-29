//
//  Parameter.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 10/18/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#ifndef LHAC_v1_Parameter_h
#define LHAC_v1_Parameter_h
#include <stdio.h>
enum loss_t {  LOG= 123, SQUARE, UNKNOWN };

struct Parameter {
    char* fileName;
    char* pfile = NULL;
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
    int dense = 1;
    
    int loss; // logistic or lasso
    
    unsigned long l; // lbfgs sy pair number <= MAX_SY_PAIRS
    int verbose;
    
    /* line search */
    int sd_flag; // 1 = sufficient decrease; 0 = line search
    
    /* gama in lbfgs */
    double shrink; // gama = gama/shrink
    
    double rho;
    
    unsigned long cd_rate;
    
    // active set stragety
    unsigned long active_set;
    bool isCached; // lasso cache aTa
    double posweight=1; // weight for pos samples
    int mdlexist = 0;
    int method_flag = 2; // 1 = ISTA; 2 = PQN - proximal quasi-newton
    
    ~Parameter() {
        delete [] fileName;
        if (pfile != NULL) delete [] pfile;
    }
    
};

#endif
