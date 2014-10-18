//
//  Lasso.cpp
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 10/18/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#include "Lasso.h"
#include "linalg.h"
#include "utils.h"

Lasso::Lasso(Parameter* param)
{
    _isCached = param->isCached;
    
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
    
    _aTb = new double[_p];
    lcdgemv(CblasColMajor, CblasTrans, Dset->X, Dset->y, _aTb, (int)_N, (int)_p, (int)_N);
    _bTb = lcddot((int)_N, Dset->y, 1, Dset->y, 1);
    if (_isCached) {
        _aTa = new double[_p*_p];
        _aTax = new double[_p];
        lcgdgemm(CblasTrans, CblasNoTrans, (int)_p, (int)_p, (int)_N,
                 1.0, Dset->X, (int)_N, Dset->X, (int)_N, 0.0, _aTa, (int)_p);

    }
    else {
        _A = new double[_p*_N];
        _Ax = new double[_N];
        memcpy(_A, Dset->X, sizeof(double)*_p*_N);
    }
}

unsigned long Lasso::getDims() const
{
    return _p;
}

double Lasso::computeObject(double* wnew)
{
    double order2 = 0;
    double order1 = 0;
    
    if (_isCached) {
        lcdgemv(CblasColMajor, CblasNoTrans, _aTa, wnew, _aTax, (int)_p, (int)_p, (int)_p);
        order2 = lcddot((int)_p, _aTax, 1, wnew, 1);
    }
    else {
        lcdgemv(CblasColMajor, CblasNoTrans, _A, wnew, _Ax, (int)_N, (int)_p, (int)_N);
        order2 = lcddot((int)_N, _Ax, 1, _Ax, 1);
        
    }
    order1 = lcddot((int)_p, _aTb, 1, wnew, 1);
    
    return 0.5*order2 - order1 + _bTb;
}

// always computed after computeObject
void Lasso::computeGradient(const double* wnew, double* df)
{
    if (_isCached) {
        for (unsigned long i = 0; i < _p; i++)
            df[i] = _aTax[i] - _aTb[i];
    }
    else {
        lcdgemv(CblasColMajor, CblasTrans, _A, _Ax, df, (int)_N, (int)_p, (int)_N);
        for (unsigned long i = 0; i < _p; i++) {
            df[i] -= _aTb[i];
        }
    }
}

Lasso::~Lasso()
{
    if (_isCached) {
        delete [] _aTa;
        delete [] _aTax;
    }
    else {
        delete [] _A;
        delete [] _Ax;
    }
    delete [] _aTb;
}

