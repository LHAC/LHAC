#include <mex.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include "sics_lhac.h"
#include "myUtilities.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2) {
        mexErrMsgIdAndTxt("LHAC:arguments",
                          "Missing arguments, please specify\n"
                          "             S - the empirical covariance matrix, and\n"
                          "             L - the regularization parameter.");
    }
    long argIdx = 0;
    // The empirical covariance matrix:
    if (!mxIsDouble(prhs[argIdx]))
        mexErrMsgIdAndTxt("LHAC:type",
                          "Expected a double matrix. (Arg. %d)",
                          argIdx + 1);
    double* S = mxGetPr(prhs[argIdx]);
    uint32_t p = (uint32_t) mxGetN(prhs[argIdx]);
    if (p != uint32_t(mxGetM(prhs[argIdx]))) {
        mexErrMsgIdAndTxt("LHAC:dimensions",
                          "Expected a square empirical covariance matrix.");
    }
    argIdx++;
    
    // Regularization parameter matrix:
    if (!mxIsDouble(prhs[argIdx]))
        mexErrMsgIdAndTxt("LHAC:type",
                          "Expected a double matrix. (Arg. %d)",
                          argIdx + 1);
    double* Lambda;
    unsigned long LambdaAlloc = 0;
    if (mxGetN(prhs[argIdx]) == 1 && mxGetM(prhs[argIdx]) == 1) {
        Lambda = (double*) malloc(p*p*sizeof(double));
        LambdaAlloc = 1;
        double lambda = mxGetPr(prhs[argIdx])[0];
        for (unsigned long i = 0; i < p*p; i++)
            Lambda[i] = lambda;
    } else {
        if (mxGetN(prhs[argIdx]) != p && mxGetM(prhs[argIdx]) != p) {
            mexErrMsgIdAndTxt("LHAC:dimensions",
                              "The regularization parameter is not a scalar\n"
                              "              or a matching matrix.");
        }
        Lambda = mxGetPr(prhs[argIdx]);
    }
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
    // data file name
    char filename[30] = {"default"};
    // max iterations
    int max_iter = 500;
    // sufficient decrease (default) or backtrack
    int sd_flag = 1;
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
    tf = mxGetField(prhs[argIdx], 0, "f");
    if (tf) {
        mxGetString(tf, filename, 30);
    }
    tf = mxGetField(prhs[argIdx], 0, "i");
    if (tf) {
        max_iter = mxGetScalar(tf);
    }
    tf = mxGetField(prhs[argIdx], 0, "l");
    if (tf) {
        sd_flag = mxGetScalar(tf);
    }
    
    
    
    param* _param = new param;
    
    
    _param->l = 10;
    _param->work_size = 1000000;
    _param->max_iter = max_iter;
    _param->max_inner_iter = 50;
    _param->opt_inner_tol = 0.05;
    _param->opt_outer_tol = opt_outer_tol;
    _param->max_linesearch_iter = 1000;
    _param->bbeta = 0.5;
    _param->ssigma = 0.001;
    _param->verbose = verbose;
    _param->sd_flag = sd_flag;
    _param->shrink = shrink;
    _param->fileName = filename;
    _param->rho = 0.01;
    _param->lmd = Lambda;
    
    solution* sols;
    sols = sics_lhac(S, p, _param);
    
//    printout("logs = ", sols, _param);
    
    double* fval = NULL;
    double* normdf =NULL;
    double* cputime = NULL;
    uint32_t* iter = NULL;

    int nlhIdx = 0;
    
    unsigned long optsize = sols->size;
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

    
    releaseSolution(sols);
    
    //    delete [] S;
    
    return;
}
