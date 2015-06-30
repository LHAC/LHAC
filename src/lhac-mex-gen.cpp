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

#define MAX_STR_LEN 200
#define NONE "none"

template <typename Derived>
Solution* optimize(const Parameter* param, double* X, double* y, uint32_t N, uint32_t p) {
    Objective<Derived>* obj = NULL;
    if (X == NULL || y == NULL) obj = new Derived(param);
    else obj = new Derived(param, X, y, N, p);
    LHAC<Derived>* Alg = new LHAC<Derived>(obj, param);
    Solution* sols = Alg->solve();
    delete obj;
    delete Alg;

    return sols;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 3 && nrhs != 1) {
        mexErrMsgIdAndTxt("LHAC:arguments",
                          "Wrong arguments, please specify\n"
                          "             param - the options struct,\n"
                          "             X - data matrix (optional when reading from param.fileName), and\n"
                          "             y - label matrix (optional when reading from param.fileName)");
    }
    
    long argIdx = 0;
    
    double* X = NULL;
    double* y = NULL;
    uint32_t N = 0;
    uint32_t p = 0;
    if (nrhs == 3) {
        argIdx = 2;
        if (!mxIsDouble(prhs[argIdx]))
            mexErrMsgIdAndTxt("LHAC:type",
                              "Expected a double matrix. (Arg. %d)",
                              argIdx + 1);
        X = mxGetPr(prhs[argIdx]);
        p = (uint32_t) mxGetN(prhs[argIdx]);
        N = (uint32_t) mxGetM(prhs[argIdx]);
        argIdx--;
        if (!mxIsDouble(prhs[argIdx]))
            mexErrMsgIdAndTxt("LHAC:type",
                              "Expected a double matrix. (Arg. %d)",
                              argIdx + 1);
        y = mxGetPr(prhs[argIdx]);
        uint32_t my = (uint32_t) mxGetM(prhs[argIdx]);
        uint32_t ny = (uint32_t) mxGetN(prhs[argIdx]);
        uint32_t maxy = (my > ny) ? my : ny;
        uint32_t miny = (my < ny) ? my : ny;
        if (N != maxy || miny != 1) {
            mexErrMsgIdAndTxt("LHAC:type",
                              "Expected a vector of length %d. (Arg. %d)",
                              N, argIdx + 1);
        }
        argIdx--;
    }
    
    // Parameter struct
    if (!mxIsStruct(prhs[argIdx])) {
        mexErrMsgIdAndTxt("LHAC:type",
                          "Expected a struct. (Arg. %d)",
                          argIdx + 1);
    }
    mxArray* tf;
    char* fileName = new char[MAX_STR_LEN];
    tf = mxGetField(prhs[argIdx], 0, "filename");
    if (!tf && nrhs == 1) {
        mexErrMsgIdAndTxt("LHAC:arguments",
                            "Wrong arguments, please specify\n"
                            "             param - the options struct,\n"
                            "             X - data matrix (optional when reading from param.fileName), and\n"
                            "             y - label matrix (optional when reading from param.fileName)");
    }
    if (tf) {
        mxGetString(tf,fileName,MAX_STR_LEN);
    }
    else {
        /* dont read from file */
        /* data pass from memory */
        strcpy(fileName, NONE);
    }
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
    unsigned long work_size = 500;
    // active set strategy -- standard (default)
//    unsigned long active_set = STD;
    unsigned long active_set = GREEDY_ADDZERO;
    int loss = LOG;
    char loss_str[MAX_STR_LEN];
    bool isCached = true;
    double lambda = 1.0;
    double posweight = 1.0;
    // LBFGS limited memory parameter
    int limited_memory = 10;
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
    tf = mxGetField(prhs[argIdx], 0, "b");
    if (tf) {
        limited_memory = mxGetScalar(tf);
    }
    tf = mxGetField(prhs[argIdx], 0, "loss");
    if (tf) {
        mxGetString(tf,loss_str,MAX_STR_LEN);
        if (strcmp(loss_str,"square")==0) loss = SQUARE;
        else if (strcmp(loss_str,"log")==0) loss = LOG;
        else loss = UNKNOWN;
    }
    tf = mxGetField(prhs[argIdx], 0, "cached");
    if (tf) {
        int b = mxGetScalar(tf);
        if (b!=0) isCached = true;
        else isCached = false;
    }
    tf = mxGetField(prhs[argIdx], 0, "lambda");
    if (tf) {
        lambda = (double) mxGetScalar(tf);
    }
    tf = mxGetField(prhs[argIdx], 0, "weight");
    if (tf) {
        posweight = (double) mxGetScalar(tf);
        if (posweight <= 0) {
            mexErrMsgIdAndTxt("LHAC:arguments",
                              "Expected a positive number for weight");
        }
    }
    
    
    Parameter* param = new Parameter;
    param->l = limited_memory;
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
    param->fileName = fileName;
    param->rho = 0.01;
    param->cd_rate = cd_rate;
    param->active_set = active_set;
    param->loss = loss;
    param->isCached = isCached;
    param->dense = 1;
    param->posweight = posweight;
    
   Solution* sols = NULL;
   switch (param->loss) {
       case SQUARE:
           printf("L1 - square\n");
           sols = optimize<Lasso>(param, X, y, N, p);
           break;
           
       case LOG:
           printf("L1 - logistic\n");
           sols = optimize<LogReg>(param, X, y, N, p);
           break;
           
       default:
           printf("Unknown loss: logistic or square!\n");
           return;
   }
    
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

