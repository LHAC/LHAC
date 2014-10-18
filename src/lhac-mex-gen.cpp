//
//  lhac-mex-gen.cpp
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 4/29/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#include <mex.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include "lhac.h"
#include "LogReg.h"
#include "Lasso.h"

template <typename Derived>
Solution* optimize(Parameter* param) {
    Objective<Derived>* obj = new Derived(param);
    //    Solution* sols = lhac(obj, param);
    LHAC<Derived>* Alg = new LHAC<Derived>(obj, param);
    Solution* sols = Alg->solve();
    delete obj;
    delete Alg;
    
    return sols;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2) {
        mexErrMsgIdAndTxt("LHAC:arguments",
                          "Missing arguments, please specify\n"
                          "             S - the empirical covariance matrix, and\n"
                          "             L - the regularization parameter.");
    }
    
    
    long argIdx = 0;
    
    // data file name
    char* filename = NULL;
    // first input is the name of the data file
    if ( mxIsChar(prhs[argIdx]) != 1)
        mexErrMsgIdAndTxt( "LHAC:inputNotString",
                          "Input must be a string.");
    filename = mxArrayToString(prhs[argIdx]);
    argIdx++;
    
    // Regularization parameter:
    if (!mxIsDouble(prhs[argIdx]))
        mexErrMsgIdAndTxt("LHAC:type",
                          "Expected a double matrix. (Arg. %d)",
                          argIdx + 1);
    double lambda = mxGetPr(prhs[argIdx])[0];
    argIdx++;
    
    // Parameter struct
    if (!mxIsStruct(prhs[argIdx])) {
        mexErrMsgIdAndTxt("LHAC:type",
                          "Expected a struct. (Arg. %d)",
                          argIdx + 1);
    }
    mxArray* tf;
    // verbose
    int verbose = 2;
    // precision
    double opt_outer_tol = 1e-6;
    // shrink -> gama = gama / shrink
    double shrink = 6;
    // max iterations
    int max_iter = 5000;
    // sufficient decrease (default) or backtrack
    int sd_flag = 1;
    // max_cdpass = 1 + iter / cdrate
    unsigned long cd_rate = 6;
    // for greedy active set
    unsigned long work_size = 300;
    // active set strategy -- standard (default)
    unsigned long active_set = STD;
    int loss = LOG;
    char* loss_str;
    bool isCached;
    tf = mxGetField(prhs[argIdx], 0, "v");
    if (tf) {
        verbose = mxGetScalar(tf);
    }
    tf = mxGetField(prhs[argIdx], 0, "e");
    if (tf) {
        opt_outer_tol = mxGetPr(tf)[0];
    }
    tf = mxGetField(prhs[argIdx], 0, "g");
    if (tf) {
        shrink = mxGetPr(tf)[0];
    }
    tf = mxGetField(prhs[argIdx], 0, "i");
    if (tf) {
        max_iter = mxGetScalar(tf);
    }
    tf = mxGetField(prhs[argIdx], 0, "l");
    if (tf) {
        sd_flag = mxGetScalar(tf);
    }
    tf = mxGetField(prhs[argIdx], 0, "r");
    if (tf) {
        cd_rate = mxGetScalar(tf);
    }
    tf = mxGetField(prhs[argIdx], 0, "w");
    if (tf) {
        work_size = mxGetScalar(tf);
    }
    tf = mxGetField(prhs[argIdx], 0, "a");
    if (tf) {
        active_set = mxGetScalar(tf);
    }
    tf = mxGetField(prhs[argIdx], 0, "loss");
    if (tf) {
        mxGetString(tf,loss_str,20);
        if (strcmp(loss_str,"square")==0) loss = SQUARE;
        if (strcmp(loss_str,"log")==0) loss = LOG;
    }
    tf = mxGetField(prhs[argIdx], 0, "cached");
    if (tf) {
        int b = mxGetScalar(tf);
        if (b!=0) isCached = true;
        else isCached = false;
    }
    
    
    Parameter* param = new Parameter;
    param->l = 10;
    param->work_size = work_size;
    param->max_iter = max_iter;
    param->lmd = lambda;
    param->max_inner_iter = 100;
    param->opt_inner_tol = 5*1e-6;
    param->opt_outer_tol = opt_outer_tol;
    param->max_linesearch_iter = 1000;
    param->bbeta = 0.5;
    param->ssigma = 0.001;
    param->verbose = verbose;
    param->sd_flag = sd_flag;
    param->shrink = shrink;
    param->fileName = filename;
    param->rho = 0.01;
    param->cd_rate = cd_rate;
    param->active_set = active_set;
    param->loss = loss;
    param->isCached = isCached;
    
    Solution* sols = NULL;
    switch (param->loss) {
        case SQUARE:
            printf("L1 - square\n");
            sols = optimize<Lasso>(param);
            break;
            
        case LOG:
            printf("L1 - logistic\n");
            sols = optimize<LogReg>(param);
            break;
            
        default:
            printf("Unknown loss: logistic or square!\n");
            return;
    }
//    LogReg* obj = new LogReg(param->fileName);
//    
//    LHAC<LogReg>* Alg = new LHAC<LogReg>(obj, param);
//    Solution* sols = Alg->solve();
    
    double* w = NULL;
    double* fval = NULL;
    double* normdf =NULL;
    double* cputime = NULL;
    uint32_t* iter = NULL;
    unsigned long* numActive = NULL;
    
    int nlhIdx = 0;
    
    unsigned long optsize = sols->size;
    if (nlhs > nlhIdx) {
        plhs[nlhIdx] = mxCreateDoubleMatrix(sols->p, 1, mxREAL);
        w = mxGetPr(plhs[nlhIdx]);
        memcpy(w, sols->w, sols->p*sizeof(double));
        nlhIdx++;
    }
    if (nlhs > nlhIdx) {
        mwSize dims[] = {optsize};
        plhs[nlhIdx] = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
        iter = (uint32_t *) mxGetData(plhs[nlhIdx]);
        memcpy(iter, sols->niter, optsize*sizeof(uint32_t));
        nlhIdx++;
    }
    if (nlhs > nlhIdx) {
        plhs[nlhIdx] = mxCreateDoubleMatrix(optsize, 1, mxREAL);
        fval = mxGetPr(plhs[nlhIdx]);
        memcpy(fval, sols->fval, optsize*sizeof(double));
        nlhIdx++;
    }
    if (nlhs > nlhIdx) {
        plhs[nlhIdx] = mxCreateDoubleMatrix(optsize, 1, mxREAL);
        cputime = mxGetPr(plhs[nlhIdx]);
        memcpy(cputime, sols->t, optsize*sizeof(double));
        nlhIdx++;
    }
    if (nlhs > nlhIdx) {
        plhs[nlhIdx] = mxCreateDoubleMatrix(optsize, 1, mxREAL);
        normdf = mxGetPr(plhs[nlhIdx]);
        memcpy(normdf, sols->normgs, optsize*sizeof(double));
        nlhIdx++;
    }
    if (nlhs > nlhIdx) {
        mwSize dims[] = {optsize};
        plhs[nlhIdx] = mxCreateNumericArray(1, dims, mxUINT64_CLASS, mxREAL);
        numActive = (unsigned long *) mxGetData(plhs[nlhIdx]);
        memcpy(numActive, sols->numActive, optsize*sizeof(unsigned long));
        nlhIdx++;
    }
    
    
    
    delete sols;
    return;
}

