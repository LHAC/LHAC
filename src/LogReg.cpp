//
//  LogReg.cpp
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 4/30/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#include "LogReg.h"
#include "linalg.h"
#include "utils.h"


LogReg::LogReg(Parameter* param)
{
    
    // sparse format
    training_set_sp* Dset_sp = new training_set_sp;
    readLibsvm(param->fileName, Dset_sp);
    
    /* statistics of the problem */
    printf("p = %ld, N = %ld, nnz = %ld\n", Dset_sp->p, Dset_sp->N, Dset_sp->nnz);
    
    training_set* Dset = new training_set;
    transformToDenseFormat(Dset, Dset_sp);
    delete Dset_sp;
    
    p = Dset->p;
    N = Dset->N;
    
    X = new double[p*N];
    y = new double[N];
    
    memcpy(X, Dset->X, sizeof(double)*p*N);
    memcpy(y, Dset->y, sizeof(double)*N);
    
    e_ywx = new double[N]; // N
    B = new double[N]; // N
    
}

unsigned long LogReg::getDims() const
{
    return p;
}

double LogReg::computeObject(double* wnew)
{
    double fval = 0;
    
//    double alpha = 1.0;
//    double beta = 0.0;
//    cblas_dgemv(CblasColMajor, CblasNoTrans, (int)N, (int)p, alpha, X, (int)N, wnew, 1, beta, e_ywx, 1);
    lcdgemv(CblasColMajor, CblasNoTrans, X, wnew, e_ywx, (int)N, (int)p, (int)N);
    for (unsigned long i = 0; i < N; i++) {
        double nc1;
        double nc2;
        nc1 = e_ywx[i]*y[i];
        e_ywx[i] = exp(nc1);
        if (nc1 <= 0) {
            nc2 = e_ywx[i];
            fval += log((1+nc2))-nc1;
        }
        else {
            nc2 = exp(-nc1);
            fval += log((1+nc2));
        }
        
    }
    
    return fval;
}

// always computed after computeObject
void LogReg::computeGradient(const double* wnew, double* df)
{
    for (unsigned long i = 0; i < N; i++) {
        B[i] = -y[i]/(1+e_ywx[i]);
    }
//    cblas_dgemv(CblasColMajor, CblasTrans, (int)N, (int)p, 1.0, X, (int)N, B, 1, 0.0, df, 1);
    lcdgemv(CblasColMajor, CblasTrans, X, B, df, (int)N, (int)p, (int)N);
    return;
}

LogReg::~LogReg()
{
    delete [] X;
    delete [] y;
    delete [] e_ywx;
    delete [] B;
}





















