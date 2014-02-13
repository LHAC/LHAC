#include <mex.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include "myUtilities.h"
#include "lhac.h"

solution* libsvmExperiment(l1log_param* param)
{
    // sparse format
    training_set_sp* Dset_sp = new training_set_sp;
    readLibsvm(param->fileName, Dset_sp);
    
    /* statistics of the problem */
    printf("p = %ld, N = %ld, nnz = %ld\n", Dset_sp->p, Dset_sp->N, Dset_sp->nnz);
    
    /* elapsed time (not cputime) */
    time_t start;
    time_t end;
    time(&start);
    double elapsedtime = 0;
    
    training_set* Dset = new training_set;
    transformToDenseFormat(Dset, Dset_sp);
    releaseProb(Dset_sp);
    
    l1log* mdl = new l1log(Dset, param);
    
    solution* sols = lhac(mdl);
    
    releaseProb(Dset);
    
    time(&end);
    elapsedtime = difftime(end, start);
    printf("%.f seconds\n", elapsedtime);
    
    delete param;
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
    int max_iter = 500;
    // sufficient decrease (default) or backtrack
    int sd_flag = 1;
    // max_cdpass = 1 + iter / cdrate
    unsigned long cd_rate = 6;
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
    
    
    l1log_param* param = new l1log_param;
    param->l = 10;
    param->work_size = 8000;
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
    
    solution* sols = NULL;
    
    sols = libsvmExperiment(param);
    
    //    printout("logs = ", sols, cparam);
    
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
