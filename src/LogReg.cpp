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


LogReg::LogReg(const Parameter* param)
{
    
    // sparse format
    training_set_sp* Dset_sp = new training_set_sp;
    readLibsvm(param->fileName, Dset_sp);
    
    /* statistics of the problem */
    printf("p = %ld, N = %ld, nnz = %ld\n", Dset_sp->p, Dset_sp->N, Dset_sp->nnz);
    
    training_set* Dset = new training_set;
    transformToDenseFormat(Dset, Dset_sp);
    delete Dset_sp;
    
    _p = Dset->p;
    _N = Dset->N;
    
    _X = new double[_p*_N];
    _y = new double[_N];
    
    memcpy(_X, Dset->X, sizeof(double)*_p*_N);
    memcpy(_y, Dset->y, sizeof(double)*_N);
//    if (param->posweight != 1.0)
//        for (unsigned long i = 0; i < _N; i++)
//            if (_y[i] >= 0) _y[i] *= param->posweight;

    _posweight = param->posweight;
    _e_ywx = new double[_N]; // N
    _B = new double[_N]; // N
    
}

LogReg::LogReg(const Parameter* param, double* X, double* y,
      unsigned long N, unsigned long p)
{
    _p = p;
    _N = N;
    
    _X = new double[_p*_N];
    _y = new double[_N];
    
    memcpy(_X, X, sizeof(double)*_p*_N);
    memcpy(_y, y, sizeof(double)*_N);
    
    _e_ywx = new double[_N]; // N
    _B = new double[_N]; // N
}

unsigned long LogReg::getDims() const
{
    return _p;
}

double LogReg::computeObject(double* wnew)
{
    double fval = 0;
    
//    double alpha = 1.0;
//    double beta = 0.0;
//    cblas_dgemv(CblasColMajor, CblasNoTrans, (int)N, (int)p, alpha, X, (int)N, wnew, 1, beta, e_ywx, 1);
    lcdgemv(CblasColMajor, CblasNoTrans, _X, wnew, _e_ywx, (int)_N, (int)_p, (int)_N);
    for (unsigned long i = 0; i < _N; i++) {
        double nc1;
        double nc2;
//        double weight = (_y[i]>0)?_posweight:1;
        nc1 = _e_ywx[i]*_y[i];
        _e_ywx[i] = exp(nc1);
        if (nc1 <= 0) {
            nc2 = _e_ywx[i];
            fval += (log((1+nc2))-nc1);
//            fval += weight*(log((1+nc2))-nc1);
        }
        else {
            nc2 = exp(-nc1);
//            fval += weight*log((1+nc2));
            fval += log((1+nc2));
        }
        
    }
    
    return fval;
}

// always computed after computeObject
void LogReg::computeGradient(const double* wnew, double* df)
{
    for (unsigned long i = 0; i < _N; i++) {
//        double weight = (_y[i]>0)?_posweight:1;
//        _B[i] = -weight*_y[i]/(1+_e_ywx[i]);
        _B[i] = -_y[i]/(1+_e_ywx[i]);
    }
//    cblas_dgemv(CblasColMajor, CblasTrans, (int)N, (int)p, 1.0, X, (int)N, B, 1, 0.0, df, 1);
    lcdgemv(CblasColMajor, CblasTrans, _X, _B, df, (int)_N, (int)_p, (int)_N);
    return;
}

LogReg::~LogReg()
{
    delete [] _X;
    delete [] _y;
    delete [] _e_ywx;
    delete [] _B;
}





















